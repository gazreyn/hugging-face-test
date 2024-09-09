// webgpu.d.ts
interface Navigator {
  gpu: GPU;
}

interface GPU {
  requestAdapter(
    options?: GPURequestAdapterOptions
  ): Promise<GPUAdapter | null>;
}

interface GPUAdapter {
  requestDevice(options?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUDevice {
  // Add any additional methods or properties you need
}
