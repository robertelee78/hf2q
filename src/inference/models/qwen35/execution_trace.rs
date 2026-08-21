//! Opaque validation of host-side GGML command-encoding observations.
//!
//! This catalog binds the exact loaded/executed weight bytes to typed
//! mlx-native 0.11.0 requests, capability decisions, and resolved Metal
//! dispatches. It proves encoding only. It does not prove command-buffer
//! submission/completion, numerical correctness, latency, or cost authority.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{
    GgmlInvocation, GgmlWorkloadClass, GGML_CAPABILITY_SCHEMA_VERSION,
    GGML_RESOLVED_DISPATCH_TRACE_SCHEMA_VERSION,
};

use super::execution_config::Qwen35ExecutionConfiguration;
use super::execution_dispatch::Qwen35EncodedDispatchObservation;
use super::execution_observation::{
    ExecutedTensorObservation, LoadedTensorCodec, VerifiedExecutedTensorCatalog,
};

const MLX_NATIVE_TRACE_VERSION: &str = "0.11.0";

pub(crate) struct VerifiedEncodedDispatchCatalog {
    executed_catalog_sha256: String,
    graph_configuration_sha256: String,
    routing_policy_sha256: String,
    observations: Vec<Qwen35EncodedDispatchObservation>,
    catalog_sha256: String,
}

impl VerifiedEncodedDispatchCatalog {
    pub(crate) fn executed_catalog_sha256(&self) -> &str {
        &self.executed_catalog_sha256
    }

    pub(crate) fn graph_configuration_sha256(&self) -> &str {
        &self.graph_configuration_sha256
    }

    pub(crate) fn routing_policy_sha256(&self) -> &str {
        &self.routing_policy_sha256
    }

    pub(crate) fn observations(&self) -> &[Qwen35EncodedDispatchObservation] {
        &self.observations
    }

    pub(crate) fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }
}

fn codec_ggml_type(codec: &LoadedTensorCodec) -> Result<GgmlType> {
    match codec {
        LoadedTensorCodec::Ggml {
            type_name,
            wire_type_id,
        } => match (type_name.as_str(), *wire_type_id) {
            ("q4_0", 2) => Ok(GgmlType::Q4_0),
            ("q8_0", 8) => Ok(GgmlType::Q8_0),
            ("q4_k", 12) => Ok(GgmlType::Q4_K),
            ("q6_k", 14) => Ok(GgmlType::Q6_K),
            _ => bail!(
                "GGML codec {type_name}/{wire_type_id} is outside the dense-Qwen trace profile"
            ),
        },
        LoadedTensorCodec::DenseF32 => {
            bail!("dense F32 execution input cannot be bound to a GGML dispatch")
        }
    }
}

fn operation_inputs<'a>(
    observation: &Qwen35EncodedDispatchObservation,
    executed: &'a VerifiedExecutedTensorCatalog,
) -> Result<Vec<&'a ExecutedTensorObservation>> {
    let inputs = observation
        .executed_tensor_node_ids
        .iter()
        .map(|node_id| {
            executed
                .observations()
                .iter()
                .find(|executed| executed.node_id == *node_id)
                .with_context(|| {
                    format!(
                        "encoded operation {} references unobserved executed node {node_id}",
                        observation.operation_id
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let expected_semantics =
        if let Some(prefix) = observation.operation_id.strip_suffix("ffn_gate_up_silu") {
            vec![
                format!("{prefix}ffn_gate.weight"),
                format!("{prefix}ffn_up.weight"),
            ]
        } else {
            vec![observation.operation_id.clone()]
        };
    if inputs
        .iter()
        .map(|input| input.semantic_name.as_str())
        .ne(expected_semantics.iter().map(String::as_str))
    {
        bail!(
            "encoded operation {} node identities do not match its stable Qwen topology",
            observation.operation_id
        );
    }
    Ok(inputs)
}

fn validate_invocation(
    observation: &Qwen35EncodedDispatchObservation,
    inputs: &[&ExecutedTensorObservation],
) -> Result<()> {
    let first = inputs
        .first()
        .context("encoded GGML operation has no executed tensor inputs")?;
    if first.shape_outermost_first.len() != 2 {
        bail!("encoded GGML weight must have a canonical two-dimensional shape");
    }
    let n = u32::try_from(first.shape_outermost_first[0])?;
    let k = u32::try_from(first.shape_outermost_first[1])?;
    let request = observation.trace.request;
    match request.invocation {
        GgmlInvocation::DenseAuto {
            m,
            n: request_n,
            k: request_k,
        } if inputs.len() == 1
            && request_n == n
            && request_k == k
            && matches!(
                (request.workload, m),
                (GgmlWorkloadClass::Prompt, 9..=u32::MAX) | (GgmlWorkloadClass::DecodeSingle, 1)
            ) => {}
        GgmlInvocation::DenseGateUpSiluPair {
            m,
            n: request_n,
            k: request_k,
        } if inputs.len() == 2
            && request_n == n
            && request_k == k
            && matches!(
                (request.workload, m),
                (GgmlWorkloadClass::Prompt, 9..=u32::MAX) | (GgmlWorkloadClass::DecodeSingle, 1)
            ) => {}
        _ => bail!(
            "encoded operation {} invocation does not match its exact executed weight inputs",
            observation.operation_id
        ),
    }
    let expected_type = codec_ggml_type(&first.executed_codec)?;
    if request.ggml_type != expected_type {
        bail!(
            "encoded operation {} GGML type differs from its executed bytes",
            observation.operation_id
        );
    }
    for input in inputs {
        if input.shape_outermost_first != first.shape_outermost_first
            || codec_ggml_type(&input.executed_codec)? != expected_type
            || input.executed_byte_len < observation.trace.capability.minimum_weight_buffer_bytes
        {
            bail!(
                "encoded operation {} has incompatible or undersized executed weight inputs",
                observation.operation_id
            );
        }
    }
    let total_bytes = inputs.iter().try_fold(0_u64, |total, input| {
        total
            .checked_add(input.executed_byte_len)
            .context("encoded operation weight-byte total overflow")
    })?;
    if observation.trace.capability.weight_buffer_count != u32::try_from(inputs.len())?
        || total_bytes < observation.trace.capability.minimum_total_weight_bytes
    {
        bail!(
            "encoded operation {} capability byte contract differs from executed inputs",
            observation.operation_id
        );
    }
    Ok(())
}

pub(super) fn verify_encoded_dispatch_catalog(
    configuration: &Qwen35ExecutionConfiguration,
    executed: &VerifiedExecutedTensorCatalog,
    observations: Vec<Qwen35EncodedDispatchObservation>,
) -> Result<VerifiedEncodedDispatchCatalog> {
    configuration.validate()?;
    if observations.is_empty() {
        bail!("encoded dispatch catalog is empty");
    }
    let expected_ggml = executed
        .observations()
        .iter()
        .filter(|observation| matches!(observation.executed_codec, LoadedTensorCodec::Ggml { .. }))
        .flat_map(|observation| {
            [
                (observation.node_id.clone(), "prompt"),
                (observation.node_id.clone(), "decode_single"),
            ]
        })
        .collect::<BTreeSet<_>>();
    let mut covered_ggml = BTreeSet::new();
    let mut observed_operations = BTreeSet::new();
    let expected_device = observations[0].trace.device.clone();
    for observation in &observations {
        let trace = &observation.trace;
        if trace.schema_version != GGML_RESOLVED_DISPATCH_TRACE_SCHEMA_VERSION
            || trace.mlx_native_version != MLX_NATIVE_TRACE_VERSION
            || trace.request.schema_version != GGML_CAPABILITY_SCHEMA_VERSION
            || trace.capability.schema_version != GGML_CAPABILITY_SCHEMA_VERSION
            || trace.capability.request != trace.request
            || trace.capability != mlx_native::ggml_capability(trace.request)
            || !trace.capability.executable
            || trace.request.routing != *configuration.ggml_routing_policy()
            || trace.device != expected_device
            || usize::try_from(trace.capability.dispatches)? != trace.dispatches.len()
            || trace.dispatches.is_empty()
        {
            bail!(
                "encoded operation {} has inconsistent typed mlx-native evidence",
                observation.operation_id
            );
        }
        if !matches!(
            trace.request.workload,
            GgmlWorkloadClass::Prompt | GgmlWorkloadClass::DecodeSingle
        ) {
            bail!(
                "encoded operation {} uses a workload outside the base text profile",
                observation.operation_id
            );
        }
        let inputs = operation_inputs(observation, executed)?;
        validate_invocation(observation, &inputs)?;
        let workload = match trace.request.workload {
            GgmlWorkloadClass::Prompt => "prompt",
            GgmlWorkloadClass::DecodeSingle => "decode_single",
            _ => unreachable!("workload was constrained above"),
        };
        if !observed_operations.insert((observation.operation_id.clone(), workload)) {
            bail!(
                "encoded operation {} was observed more than once for workload {workload}",
                observation.operation_id
            );
        }
        for input in inputs {
            covered_ggml.insert((input.node_id.clone(), workload));
        }
    }
    if covered_ggml != expected_ggml {
        bail!(
            "encoded GGML coverage differs from executed catalog: expected {:?}, observed {:?}",
            expected_ggml,
            covered_ggml
        );
    }
    let graph_configuration_sha256 = configuration.graph_configuration_sha256().to_owned();
    let routing_policy_sha256 = configuration.routing_policy_sha256().to_owned();
    let catalog_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(&(
        executed.catalog_sha256(),
        &graph_configuration_sha256,
        &routing_policy_sha256,
        &observations,
    ))?));
    Ok(VerifiedEncodedDispatchCatalog {
        executed_catalog_sha256: executed.catalog_sha256().to_owned(),
        graph_configuration_sha256,
        routing_policy_sha256,
        observations,
        catalog_sha256,
    })
}
