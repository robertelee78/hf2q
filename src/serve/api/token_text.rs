//! Raw vocabulary bytes and UTF-8-safe incremental token text (#553).
//!
//! `Tokenizer::decode(&[id])` is lossy for tokens that carry only part of a
//! UTF-8 code point: a byte-fallback piece such as Gemma's `<0x82>`, or a
//! byte-level BPE piece holding an incomplete sequence, renders as U+FFFD.
//! The grammar mask and logprobs need the bytes the token really contributes,
//! and per-token text emission must hold an incomplete code point back until
//! the token that completes it arrives.

use tokenizers::decoders::DecoderWrapper;
use tokenizers::Tokenizer;

/// The bytes token `token` contributes to the output stream.
///
/// Equal to `tokenizer.decode(&[token], false)` whenever that rendering is
/// lossless. When it contains U+FFFD and the decoder chain maps the
/// vocabulary piece to raw bytes (`ByteFallback` `<0xHH>` pieces or
/// `ByteLevel` pieces), those raw bytes are returned instead, so a lone
/// continuation byte stays the invalid byte it is rather than the valid
/// character U+FFFD. A piece that literally spells U+FFFD keeps its text.
pub(crate) fn token_raw_bytes(tokenizer: &Tokenizer, token: u32) -> Vec<u8> {
    let text = tokenizer.decode(&[token], false).unwrap_or_default();
    if !text.contains(char::REPLACEMENT_CHARACTER) {
        return text.into_bytes();
    }
    raw_piece_bytes(tokenizer, token).unwrap_or_else(|| text.into_bytes())
}

fn raw_piece_bytes(tokenizer: &Tokenizer, token: u32) -> Option<Vec<u8>> {
    let decoder = tokenizer.get_decoder()?;
    let piece = tokenizer.id_to_token(token)?;
    if decoder_contains(decoder, &|d| matches!(d, DecoderWrapper::ByteFallback(_))) {
        // Mirrors `ByteFallback::decode_chain`: exactly `<0xHH>`.
        if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
            if let Ok(byte) = u8::from_str_radix(&piece[3..5], 16) {
                return Some(vec![byte]);
            }
        }
        return None;
    }
    if decoder_contains(decoder, &|d| matches!(d, DecoderWrapper::ByteLevel(_))) {
        // Mirrors `ByteLevel::decode_chain`: every char maps to one byte,
        // otherwise the piece contributes its own UTF-8 bytes.
        return piece.chars().map(byte_level_byte).collect();
    }
    None
}

fn decoder_contains(decoder: &DecoderWrapper, wanted: &dyn Fn(&DecoderWrapper) -> bool) -> bool {
    match decoder {
        DecoderWrapper::Sequence(sequence) => sequence
            .get_decoders()
            .iter()
            .any(|inner| decoder_contains(inner, wanted)),
        other => wanted(other),
    }
}

/// Inverse of GPT-2 `bytes_to_unicode`: printable bytes map to themselves;
/// the 68 others map, in ascending byte order, to U+0100..=U+0143.
fn byte_level_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    match cp {
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => Some(cp as u8),
        0x100..=0x120 => Some((cp - 0x100) as u8),
        0x121..=0x142 => Some((cp - 0x121 + 0x7F) as u8),
        0x143 => Some(0xAD),
        _ => None,
    }
}

/// Incremental token-to-text decoder that never splits a UTF-8 code point.
///
/// `push` returns the text that became final with this token: tokens whose
/// bytes start a multi-byte character yield nothing until the character is
/// complete, and bytes that can never form a character become U+FFFD exactly
/// as `String::from_utf8_lossy` would render them. Unary text is the
/// concatenation of every `push` result plus `finish`; SSE emits each `push`
/// result as its fragment.
#[derive(Debug, Default)]
pub(crate) struct TokenTextDecoder {
    pending: Vec<u8>,
}

impl TokenTextDecoder {
    pub(crate) fn push(&mut self, tokenizer: &Tokenizer, token: u32) -> String {
        let bytes = token_raw_bytes(tokenizer, token);
        if self.pending.is_empty() {
            match String::from_utf8(bytes) {
                Ok(text) => return text,
                Err(error) => self.pending = error.into_bytes(),
            }
        } else {
            self.pending.extend_from_slice(&bytes);
        }
        let mut text = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    text.push_str(valid);
                    self.pending.clear();
                    return text;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    text.push_str(
                        std::str::from_utf8(&self.pending[..valid])
                            .expect("prefix up to valid_up_to is UTF-8"),
                    );
                    match error.error_len() {
                        None => {
                            self.pending.drain(..valid);
                            return text;
                        }
                        Some(invalid) => {
                            text.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + invalid);
                        }
                    }
                }
            }
        }
    }

    /// Flush a held-back incomplete sequence at end of generation.
    pub(crate) fn finish(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }

    /// Drop held-back bytes when the caller replaces or truncates the text
    /// (stop string, grammar-dead re-decode).
    pub(crate) fn discard(&mut self) {
        self.pending.clear();
    }
}

#[cfg(test)]
#[path = "token_text_tests.rs"]
mod tests;
