use super::*;
pub(super) const WIDTH: usize=5120;
pub(super) const LAYERS: usize=64;
pub(super) const POINTS: usize=1+2*LAYERS;
pub(crate) const ARM_BYTES: u64=(2*POINTS*WIDTH*4) as u64;
pub(crate) fn expected() -> Vec<ExpectedTensor> {
    let mut out=Vec::new();
    for position in 129..131 {
        out.push(item(position,0,Stage::Input));
        for layer in 0..LAYERS as u32 {
            out.push(item(position,layer,Stage::Residual));
            out.push(item(position,layer,Stage::LayerOutput));
        }
    }
    out
}
fn item(position: usize,layer: u32,stage: Stage)->ExpectedTensor {
    ExpectedTensor {at:at(position,layer,stage),scalar:Scalar::F32,shape:vec![1,WIDTH]}
}
pub(super) fn at(position: usize,layer: u32,stage: Stage)->Coordinate {
    Coordinate {logical_step:position as u64,layer,stage,agent:"agent-0".into(),
        logical_row_start:0,segment:0}
}
pub(super) fn point(layer: usize,stage: Stage)->Result<usize> {
    anyhow::ensure!(layer<LAYERS,"selected layer range");
    match stage {
        Stage::Input if layer==0=>Ok(0),
        Stage::Residual=>Ok(1+2*layer),
        Stage::LayerOutput=>Ok(2+2*layer),
        _=>anyhow::bail!("unsupported target checkpoint"),
    }
}
pub(super) fn select(forced: bool,tokens:&[u32],positions:&[i32],wanted:[u32;2])
    ->Result<Option<Vec<usize>>> {
    let n=tokens.len();
    if positions.first().is_none_or(|&p|p!=129 && p!=130) {return Ok(None)}
    anyhow::ensure!(positions.len()==4*n && (0..4).all(|axis|
        (0..n).all(|row|positions[axis*n+row]==positions[0]+row as i32)),
        "actual axis-major positions; no physical-row inference");
    if forced {
        anyhow::ensure!(n==4 && positions[0]==129 && tokens[..2]==wanted,
            "actual retained R4 first-two token inputs");
        Ok(Some(vec![129,130]))
    } else {
        let position=positions[0] as usize;
        anyhow::ensure!(n==1 && tokens[0]==wanted[position-129],"actual retained scalar input");
        Ok(Some(vec![position]))
    }
}

pub(super) fn visit(seen:&mut BTreeSet<usize>,layer:usize,stage:Stage)->Result<usize> {
    let point=point(layer,stage)?;
    anyhow::ensure!(seen.insert(point),"duplicate target checkpoint");Ok(point)
}
pub(super) fn complete(seen:&BTreeSet<usize>)->Result<()> {
    anyhow::ensure!(seen.iter().copied().eq(0..POINTS),"required layer checkpoint omitted/replaced");Ok(())
}
