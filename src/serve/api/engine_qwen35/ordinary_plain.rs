//! Existing ordinary scalar head/sampler separated at its target boundary.
use super::*;

pub(crate) struct PreparedPlainTarget {
    pub(super) token: u32,
    pub(super) position: i32,
    slot: SlotId,
    step: usize,
    generated_len: usize,
    prior_target: u32,
    forced_token: Option<(u32, bool)>,
    greedy: bool,
}
impl PreparedPlainTarget {
    pub(in crate::serve::api) fn is_greedy(&self) -> bool { self.greedy }
    pub(crate) fn input(&self) -> (u32, i32, SlotId) { (self.token,self.position,self.slot) }
}
pub(crate) enum PlainTargetResult { Greedy(u32), FullLogits(Vec<f32>) }
impl Qwen35DecodeState {
    pub(super) fn decode_tick_plain_ordinary(&mut self, qwen:&mut Qwen35LoadedModel,
        kv:&mut HybridKvCache, supervisor:&EngineSupervisor, greedy:bool) -> Result<Qwen35TickOutcome> {
        let started=Instant::now();
        let prepared=self.prepare_plain_target(kv,greedy)?;
        let target=self.produce_scalar_plain_target(qwen,kv,supervisor,&prepared)?;
        self.finish_plain_target(qwen,kv,prepared,target,started)
    }
    pub(in crate::serve::api) fn prepare_plain_target(&mut self, kv:&HybridKvCache,
        greedy:bool) -> Result<PreparedPlainTarget> {
        let token=*self.generated_tokens.last().context("ordinary target needs seed")?;
        let position=(self.decode_position_base+self.step-1) as i32;
        let forced_token=self.thinking_budget.as_mut().and_then(Qwen35ThinkingBudgetState::next_forced_token);
        if forced_token.is_some_and(|(_,started)|started) {
            tracing::warn!(slot=self.slot_id.0,budget=self.params.thinking_token_budget,
                generated_tokens=self.generated_tokens.len(),
                "Qwen35 thinking token budget reached; forcing reasoning close and continuing answer");
        }
        Ok(PreparedPlainTarget { token,position,slot:self.slot_id,step:self.step,
            generated_len:self.generated_tokens.len(),prior_target:kv.sequence_len_for_slot(self.slot_id)?,
            forced_token,greedy })
    }
    pub(in crate::serve::api) fn produce_scalar_plain_target(&mut self,qwen:&mut Qwen35LoadedModel,
        kv:&mut HybridKvCache,supervisor:&EngineSupervisor,p:&PreparedPlainTarget) -> Result<PlainTargetResult> {
        if p.greedy {
            let lease=supervisor.arm("Qwen35 decode greedy",QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
            let forward=qwen.model.forward_gpu_greedy(&[p.token],&[p.position;4],kv,self.slot_id);
            lease.finish()?;
            Ok(PlainTargetResult::Greedy(forward.with_context(||format!(
                "Qwen35Model::forward_gpu_greedy (slot-aware decode step {}; ADR-040 Phase F M1)",self.step))?))
        } else {
            let lease=supervisor.arm("Qwen35 decode logits",QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
            let forward=qwen.model.forward_gpu_last_logits(&[p.token],&[p.position;4],kv,self.slot_id);
            lease.finish()?;
            let logits=forward.with_context(||format!("Qwen35Model::forward_gpu_last_logits (slot-aware decode step {})",self.step))?;
            anyhow::ensure!(logits.len()==qwen.vocab_size,"qwen35 slot-aware decode logits len {} != vocab_size {}",logits.len(),qwen.vocab_size);
            Ok(PlainTargetResult::FullLogits(logits))
        }
    }
    pub(in crate::serve::api) fn validate_plain_result(&self,kv:&HybridKvCache,
        p:&PreparedPlainTarget,target:&PlainTargetResult,vocab:usize) -> Result<()> {
        anyhow::ensure!(self.slot_id==p.slot&&self.step==p.step&&self.generated_tokens.len()==p.generated_len
            &&self.generated_tokens.last()==Some(&p.token)
            &&(self.decode_position_base+self.step-1) as i32==p.position,"stale ordinary target state");
        anyhow::ensure!(kv.sequence_len_for_slot(p.slot)?==p.prior_target.checked_add(1).context("ordinary cursor overflow")?,"ordinary target must advance once");
        anyhow::ensure!(matches!((p.greedy,target),(true,PlainTargetResult::Greedy(_)))
            ||matches!((p.greedy,target),(false,PlainTargetResult::FullLogits(v))if v.len()==vocab),"ordinary head/result mismatch");
        Ok(())
    }
    pub(in crate::serve::api) fn finish_plain_target(&mut self,qwen:&mut Qwen35LoadedModel,
        kv:&HybridKvCache,p:PreparedPlainTarget,target:PlainTargetResult,ordinary_target_started:Instant) -> Result<Qwen35TickOutcome> {
        self.finish_plain_with_cost(qwen,kv,p,target,
            super::ordinary_cost::OrdinaryCost::OriginalScalar(ordinary_target_started))
    }
    pub(in crate::serve::api) fn finish_plain_with_cost(&mut self,qwen:&mut Qwen35LoadedModel,
        kv:&HybridKvCache,p:PreparedPlainTarget,target:PlainTargetResult,
        cost:super::ordinary_cost::OrdinaryCost)->Result<Qwen35TickOutcome> {
        let own_tail_started=Instant::now();
        self.validate_plain_result(kv,&p,&target,qwen.vocab_size)?;
        let forced_token=p.forced_token;
        let tok=match target {
            PlainTargetResult::Greedy(predicted)=>forced_token.map_or(predicted,|(token,_)|token),
            PlainTargetResult::FullLogits(mut logits)=>{
                let (token,logprob)=if let Some((token,_))=forced_token {
                    let record=TokenLogprobRecord {token_id:token,logprob:0.0,top:Vec::new()};
                    (token,self.want_logprobs.then_some(record))
                } else { sample_logits_qwen35_constrained(&mut logits,&self.params,&self.sampling_history,
                    self.grammar_runtime.as_ref(),&qwen.eos_token_ids,self.want_logprobs)? };
                if let (Some(values),Some(logprob))=(self.logprobs_vec.as_mut(),logprob) { values.push(logprob); }
                token
            }
        };
        let ordinary_target_elapsed = cost.complete(own_tail_started);
        self.mtp_cost
            .observe_ordinary_target(ordinary_target_elapsed);
        self.history_cost
            .observe_ordinary_target(ordinary_target_elapsed);
        advance_qwen35_grammar(
            &mut self.grammar_runtime,
            &self.params,
            &qwen.eos_token_ids,
            tok,
        )?;
        if qwen.eos_token_ids.contains(&tok) {
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            });
        }
        if qwen35_grammar_terminal_token(self.grammar_runtime.as_ref(), &self.params, tok) {
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            });
        }
        self.generated_tokens.push(tok);
        qwen35_observe_sampling_history(&mut self.sampling_history, tok);
        let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
        self.decoded_text.push_str(&frag);
        let mut tool_opened = false;
        if let Some(splitter) = self.tool_splitter.as_mut() {
            let marker_events = splitter.feed(&frag);
            tool_opened = marker_events
                .iter()
                .any(|event| matches!(event, ToolCallEvent::ToolCallOpen));
            if tool_opened {
                if let Some(runtime) = self.grammar_runtime.as_mut() {
                    runtime.trigger();
                }
            }
        }
        if let Some(budget) = self.thinking_budget.as_mut() {
            budget.observe_generated(&self.generated_tokens, tool_opened);
        }
        if qwen35_hit_stop_string(&self.decoded_text, &self.stop_strings) {
            qwen35_strip_trailing_stop(&mut self.decoded_text, &self.stop_strings);
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: frag,
                is_reasoning: false,
                finished: true,
            });
        }
        self.step += 1;
        let finished = self.step >= self.max_tokens;
        Ok(Qwen35TickOutcome {
            fragment: frag,
            is_reasoning: false,
            finished,
        })
    }
}
