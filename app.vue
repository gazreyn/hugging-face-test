<script lang="ts" setup>
import {
  env,
  AutoModel,
  AutoProcessor,
  Processor,
  RawImage,
  type PreTrainedModel,
} from '@huggingface/transformers';

import JSZip from 'jszip';
import { saveAs } from 'file-saver';

const fileInput = ref<HTMLInputElement | null>(null);
const modelRef = ref<PreTrainedModel | null>(null);
const processorRef = ref<Processor | null>(null);
const images = ref<any>([]);
const processedImages = ref<any>([]);
const isLoading = ref(false);
const isProcessing = ref(false);
const isDownloadReady = ref(false);
//
const error = ref('');

onMounted(async () => {
  try {
    isLoading.value = true;

    if (!window.navigator.gpu) {
      throw new Error('WebGPU not supported');
    }

    const model_id = 'briaai/RMBG-1.4';
    // @ts-ignore
    env.backends.onnx.wasm.proxy = false;
    modelRef.value ??= await AutoModel.from_pretrained(model_id, {
      device: 'webgpu',
    });

    processorRef.value ??= await AutoProcessor.from_pretrained(model_id);
  } catch (error: any) {
    error.value = error;
  } finally {
    isLoading.value = false;
  }
});

const removeImage = (index: number) => {
  images.value.splice(index, 1);
  processedImages.value.splice(index, 1);
};

const processImage = async () => {
  isProcessing.value = true;
  processedImages.value = [];

  const model = modelRef.value!;
  const processor = processorRef.value!;

  for (let i = 0; i < images.value.length; i++) {
    // Load Image
    const img = await RawImage.fromURL(images.value[i]);

    // Pre-process Image
    const { pixel_values } = await processor(img);

    // Predict Alpha Matte
    const { output } = await model({ input: pixel_values });

    const maskData = (
      await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(
        img.width,
        img.height
      )
    ).data;

    // Create a new Canvas
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d')!;

    // Draw the image
    ctx.drawImage(img.toCanvas(), 0, 0);

    // Update alpha channel
    const pixelData = ctx.getImageData(0, 0, img.width, img.height);
    for (let i = 0; i < maskData.length; ++i) {
      pixelData.data[4 * i + 3] = maskData[i];
    }
    ctx.putImageData(pixelData, 0, 0);
    processedImages.value.push(canvas.toDataURL('image/png'));
  }

  isProcessing.value = false;
  isDownloadReady.value = true;
};

const downloadAsZip = async () => {
  const zip = new JSZip();

  const promises = images.value.map((image: string, index: number) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      const img = new Image();
      img.src = processedImages.value[index] || image;

      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx?.drawImage(img, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) {
            zip.file(`image-${index + 1}.png`, blob);
          }
          resolve(null);
        }, 'image/png');
      };
    });
  });

  await Promise.all(promises);

  const content = await zip.generateAsync({ type: 'blob' });
  saveAs(content, 'images.zip');
};

const copyToClipboard = async (url: string) => {
  try {
    const response = await fetch(url);
    const blob = await response.blob();

    // Create a clipboard item with the image
    const clipboardItem = new ClipboardItem({
      [blob.type]: blob,
    });

    // Write the clipboard item to the clipboard
    await navigator.clipboard.write([clipboardItem]);

    console.log('Image copied to clipboard');
  } catch (error) {
    console.error('Failed to copy image to clipboard:', error);
  }
};

const downloadImage = (url: string) => {
  const link = document.createElement('a');
  link.href = url;
  link.download = 'image.png';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

function handleImagesSelect(event: Event) {
  const files = (event.target as HTMLInputElement).files;
  if (!files) return;

  images.value = Array.from(files).map((img) => {
    return URL.createObjectURL(img);
  });
}

const clearAll = () => {
  images.value = [];
  processedImages.value = [];
  isDownloadReady.value = false;
  fileInput.value && (fileInput.value.value = '');
};
</script>

<template>
  <div>
    <h1>Hugging Face Test</h1>
    <input
      type="file"
      multiple
      accept="image/*"
      @change="handleImagesSelect"
      ref="fileInput"
    />
    <button @click="processImage">Process</button>
    <button @click="downloadAsZip">Download as Zip</button>
    <button @click="clearAll">Clear</button>
    <ul>
      <li
        v-for="(image, index) in images"
        :key="index"
      >
        <img :src="processedImages[index] || image" /> <br />
        <button @click="removeImage(index)">Remove</button>
        <button @click="copyToClipboard(processedImages[index])">
          Copy to Clipboard
        </button>
        <button @click="downloadImage(processedImages[index])">Download</button>
      </li>
    </ul>
  </div>
</template>
