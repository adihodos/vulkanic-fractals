use crate::vulkan_renderer::GraphicsError;

pub struct UniqueBufferMapping<'a> {
    buffer_mem: ash::vk::DeviceMemory,
    pub mapped_memory: *mut std::ffi::c_void,
    device: &'a ash::Device,
    pub offset: usize,
    pub size: usize,
}

impl<'a> UniqueBufferMapping<'a> {
    pub fn map_memory(
        device: &'a ash::Device,
        device_memory: ash::vk::DeviceMemory,
        offset: usize,
        map_size: usize,
    ) -> std::result::Result<Self, GraphicsError> {
        let mapped_memory = unsafe {
            device.map_memory(
                device_memory,
                offset as u64,
                map_size as u64,
                ash::vk::MemoryMapFlags::empty(),
            )
        }?;

        Ok(Self {
            buffer_mem: device_memory,
            mapped_memory,
            device,
            offset,
            size: map_size,
        })
    }

    // pub fn new(
    //     buf: &VulkanBuffer,
    //     ds: &'a VulkanDeviceState,
    //     offset: Option<usize>,
    //     size: Option<usize>,
    // ) -> std::result::Result<Self, GraphicsError> {
    //     Self::map_memory(
    //         &ds.device,
    //         buf.memory,
    //         offset.unwrap_or(0),
    //         size.unwrap_or(ash::vk::WHOLE_SIZE as usize),
    //     )
    // }

    pub fn write_data<T: Sized + Copy>(&self, data: &[T]) {
        unsafe {
            let dst = (self.mapped_memory as *mut u8).offset(self.offset as isize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut T, data.len());
        }
    }

    pub fn write_data_with_offset<T: Sized + Copy>(&mut self, data: &[T], offset: isize) {
        unsafe {
            let dst = (self.mapped_memory as *mut u8).offset(self.offset as isize);
            let dst = (dst as *mut T).offset(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut T, data.len());
        }
    }
}

impl<'a> std::ops::Drop for UniqueBufferMapping<'a> {
    fn drop(&mut self) {
        unsafe {
            let _ = self
                .device
                .flush_mapped_memory_ranges(&[ash::vk::MappedMemoryRange::default()
                    .memory(self.buffer_mem)
                    .offset(self.offset as ash::vk::DeviceSize)
                    .size(self.size as ash::vk::DeviceSize)]);
            self.device.unmap_memory(self.buffer_mem);
        }
    }
}
