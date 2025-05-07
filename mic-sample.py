import numpy as np
import onnxruntime
import soundfile as sf  # type:ignore
import sounddevice as sd  # type:ignore


# 入力用コールバック
def audio_callback(indata, outdata, frames, time, status):
    global input_buffer
    input_buffer = np.concatenate((input_buffer, indata[:, 0]))


# パラメータ
window_size = 512
hop_size = 256
n_fft = 512
sampling_rate = 16000
window = np.sqrt(np.hanning(window_size))

# バッファ
input_buffer = np.zeros(0, dtype=np.float32)
temp_output_buffer = np.zeros(window_size, dtype=np.float32)
output_segments = []

# モデル・状態
model = onnxruntime.InferenceSession(
    "onnx_model/gtcrn_simple.onnx", providers=["CPUExecutionProvider"]
)
conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

# ストリーム開始
with sd.Stream(
    samplerate=sampling_rate,
    blocksize=1024,
    dtype="float32",
    channels=1,
    callback=audio_callback,
):
    print("Real-time GTCRN enhancement... [Press Ctrl+C to stop]")

    try:
        while True:
            while len(input_buffer) >= window_size:
                # チャンク取り出し
                chunk = input_buffer[:window_size]
                input_buffer = input_buffer[hop_size:]

                # フーリエ変換
                chunk_windowed = chunk * window
                chunk_spec = np.fft.rfft(chunk_windowed, n=n_fft)
                real = np.real(chunk_spec)
                imag = np.imag(chunk_spec)
                input_data = np.stack([real, imag], axis=-1)[None, :, None, :]

                # 推論
                output_data, conv_cache, tra_cache, inter_cache = model.run(
                    None,
                    {
                        "mix": input_data.astype(np.float32),
                        "conv_cache": conv_cache,
                        "tra_cache": tra_cache,
                        "inter_cache": inter_cache,
                    },
                )

                # 逆実数フーリエ変換
                out_real = output_data[0][:, 0, 0]
                out_imag = output_data[0][:, 0, 1]
                enhanced_spec = out_real + 1j * out_imag
                time_chunk = np.fft.irfft(enhanced_spec, n=n_fft)[:window_size]
                time_chunk *= window

                # オーバーラップ加算
                temp_output_buffer += time_chunk
                output_segments.append(temp_output_buffer[:hop_size].copy())

                # シフト
                temp_output_buffer = np.roll(temp_output_buffer, -hop_size)
                temp_output_buffer[-hop_size:] = 0.0

            sd.sleep(1)

    except KeyboardInterrupt:
        print("Saving output to 'output.wav'...")
        final_output = np.concatenate(output_segments)
        sf.write("output.wav", final_output, samplerate=sampling_rate)
