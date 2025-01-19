use std::sync::atomic::Ordering;

/// A spin mutex.
/// adapted from https://github.com/marzer/spin_mutex
pub struct SpinMutex {
    held: std::sync::atomic::AtomicBool,
}

impl SpinMutex {
    pub fn new() -> SpinMutex {
        SpinMutex {
            held: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn lock(&self) -> UniqueSpinMutexLock {
        let mut mask = 1;
        const MAX_BACKOFF_COUNT: i32 = 64;

        while let Err(_) =
            self.held
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        {
            while self.held.load(std::sync::atomic::Ordering::Relaxed) {
                let mut i = mask;
                while i > 0 {
                    std::hint::spin_loop();
                    i -= 1;
                }
                mask = if mask < MAX_BACKOFF_COUNT {
                    mask << 1
                } else {
                    MAX_BACKOFF_COUNT
                };
            }
        }

        UniqueSpinMutexLock::with_mutex(self)
    }

    pub fn unlock(&self) {
        self.held.store(false, Ordering::Release)
    }
}

pub struct UniqueSpinMutexLock<'a> {
    mutex: &'a SpinMutex,
}

impl<'a> UniqueSpinMutexLock<'a> {
    fn with_mutex(mutex: &'a SpinMutex) -> Self {
        Self { mutex }
    }
}

impl<'a> std::ops::Drop for UniqueSpinMutexLock<'a> {
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}
