//! A retained legacy pool belongs to one verifier device/queue. A checked
//! reset retires every issued use; only then may another device replace it.

use std::cell::RefCell;

use mlx_native::{MlxBufferPool, MlxDevice, MlxError};

thread_local! {
    // Retaining the device also prevents queue-pointer reuse while it is bound.
    static OWNER: RefCell<Option<MlxDevice>> = const { RefCell::new(None) };
}

#[inline]
pub(super) fn bind(device: &MlxDevice) -> Result<(), MlxError> {
    OWNER.with(|owner| {
        let mut owner = owner.borrow_mut();
        if let Some(previous) = owner.as_ref() {
            if std::ptr::eq(previous.metal_device(), device.metal_device())
                && std::ptr::eq(&**previous.metal_queue(), &**device.metal_queue())
            {
                return Ok(());
            }
            // These are established only by the caller's checked drain/reset.
            // Never retire an active cycle or turn a failed drain into reuse.
            if super::issued_bytes() != 0
                || super::DECODE_POOL.with(|pool| pool.borrow().in_use_count() != 0)
                || super::OVERFLOW.with(|overflow| !overflow.borrow().is_empty())
            {
                return Err(MlxError::InvalidArgument(
                    "Qwen scratch device changed without a checked pool reset".into(),
                ));
            }
            super::drop_id_mm_scratch();
            super::DECODE_POOL.with(|pool| pool.replace(MlxBufferPool::new()));
            super::FREE_AT_RESET.with(|free| free.set(0));
            super::POOLED_SINCE_RESET.with(|pooled| pooled.set(0));
        }
        *owner = Some(device.clone());
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::DType;

    #[test]
    fn another_device_requires_a_checked_reset_and_gets_a_fresh_domain() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(first) = MlxDevice::new() else { return };
        let Ok(second) = MlxDevice::new() else { return };
        let mut current = super::super::pooled_alloc_buffer(&first, 4, DType::U32, vec![1]).unwrap();
        current.as_mut_slice::<u32>().unwrap()[0] = 42;
        assert!(matches!(
            super::super::pooled_alloc_buffer(&second, 4, DType::U32, vec![1]),
            Err(MlxError::InvalidArgument(_))
        ));
        assert_eq!(current.as_slice::<u32>().unwrap(), &[42]);
        drop(current);
        super::super::checked_reset_decode_pool().unwrap();

        // Exhaust the formerly free bucket, forcing a fresh native allocation
        // too. Merely reusing one old-domain root would hide the binding bug.
        let mut a = super::super::pooled_alloc_buffer(&second, 4, DType::U32, vec![1]).unwrap();
        let mut b = super::super::pooled_alloc_buffer(&second, 4, DType::U32, vec![1]).unwrap();
        a.as_mut_slice::<u32>().unwrap()[0] = 7;
        b.as_mut_slice::<u32>().unwrap()[0] = 13;
        assert_eq!(a.as_slice::<u32>().unwrap(), &[7]);
        assert_eq!(b.as_slice::<u32>().unwrap(), &[13]);
        drop((a, b));
        super::super::checked_reset_decode_pool().unwrap();
    }
}
