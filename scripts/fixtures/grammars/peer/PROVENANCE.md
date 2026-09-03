# llama.cpp GBNF fixture provenance

These eight `.gbnf` files are copied verbatim from the `grammars/` directory
of the local clean llama.cpp checkout at:

- Repository: <https://github.com/ggml-org/llama.cpp>
- Commit: `9cffdcc801582616250520966699cb5b25d28243`
- Commit date: 2026-09-02
- Captured: 2026-09-03

hf2q's `data/llama_cpp_pin.txt` names
`e15384a5cb092b080c2a01c0b9e3f8635079d6df`. That commit was not available
in the local llama.cpp object database when these fixtures were captured, so
these files MUST NOT be represented as fixtures from the repository pin.

The fixture files and their SHA-256 digests are:

| File | SHA-256 |
| --- | --- |
| `arithmetic.gbnf` | `dd91e1ef7bb0178e398036c6f5251c694b79c43cb2d223784f4dcf3020994776` |
| `c.gbnf` | `556cdbde93b832c9283896d68837fd3025e772677b36dc18f066b194fb99698c` |
| `chess.gbnf` | `de1a656bb4cf0fda3991f0d3f1bcde338c0d82a60f7e0e8cabb2ba0c73002e5f` |
| `english.gbnf` | `96a248f563c5780af621b63d59c64e8d4823cb3b9b3e7f487eb8f8b3d11b0dcf` |
| `japanese.gbnf` | `69fc177180f2b15d085d811aa41e360bb85c962d424d7c17dc936c04ca668e00` |
| `json.gbnf` | `1a3e3469892957861029adcd24ad0c886b778a942dd4a18cc146238ad56474dd` |
| `json_arr.gbnf` | `f817496d5ee08aaa7dac4fdee59cbecfcf99367088b63e4b1c02d6f632c39206` |
| `list.gbnf` | `7e8b142784f991511650e5f0e82d18a123558f09e9d19a056d10718a6986a150` |

The source is MIT-licensed. `LICENSE.llama.cpp` is the verbatim license from
the same commit and MUST remain with these vendored fixtures.
