import onnxruntime as ort
import numpy as np
import soundfile as sf
import os

class FixedVADTest:
    def __init__(self):
        # 加载VAD模型
        vad_model_path = "silero-vad/onnx/model.onnx"
        self.session = ort.InferenceSession(vad_model_path)
        
        # 设置VAD参数
        self.sample_rate = 16000
        self.window_size = 512  # 32ms @ 16kHz
        
    def get_model_input_details(self):
        """获取模型输入详情"""
        # 内部方法，不再打印详细信息
        pass
            
    def preprocess_audio(self, audio_path):
        """预处理音频"""
        audio_data, sr = sf.read(audio_path, dtype='float32')
        
        # 转换为单声道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 重采样
        if sr != self.sample_rate:
            from resampy import resample
            audio_data = resample(audio_data, sr, self.sample_rate)
        
        # 归一化
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        return audio_data
    
    def detect_voice_segments(self, audio_data):
        """检测语音段"""
        # 修复：状态形状应该是(2, 1, 128)而不是(2, 1, 64)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        
        # 分帧处理
        audio_length = len(audio_data)
        num_windows = int(np.ceil(audio_length / self.window_size))
        
        # 存储语音段（先不做扩展）
        raw_voice_segments = []
        vad_probs = []
        in_voice = False
        segment_start = 0
        
        # 获取模型输入名称
        input_names = [input.name for input in self.session.get_inputs()]
        
        for i in range(num_windows):
            start = i * self.window_size
            end = min(start + self.window_size, audio_length)
            window = audio_data[start:end]
            
            # 填充不足一帧的数据
            if len(window) < self.window_size:
                window = np.pad(window, (0, self.window_size - len(window)), 'constant')
            
            # 准备输入
            ort_inputs = {}
            if 'input' in input_names:
                ort_inputs['input'] = window.reshape(1, -1).astype(np.float32)
            if 'state' in input_names:
                ort_inputs['state'] = state
            if 'sr' in input_names:
                ort_inputs['sr'] = np.array([self.sample_rate], dtype=np.int64)
            
            # 推理
            try:
                ort_outs = self.session.run(None, ort_inputs)
                vad_prob = float(ort_outs[0][0][0])
                vad_probs.append(vad_prob)
                
                # 更新状态
                if len(ort_outs) > 1:
                    state = ort_outs[1]
                
                # 检测语音段
                threshold = 0.15
                if vad_prob > threshold:
                    if not in_voice:
                        in_voice = True
                        segment_start = start
                else:
                    if in_voice:
                        in_voice = False
                        # 只添加有意义长度的语音段
                        if start - segment_start > self.sample_rate * 0.1:  # 至少100ms
                            raw_voice_segments.append({"start": segment_start, "end": start})
            except Exception as e:
                vad_probs.append(0.0)
        
        # 处理最后一个语音段
        if in_voice and segment_start < audio_length:
            if audio_length - segment_start > self.sample_rate * 0.1:
                raw_voice_segments.append({"start": segment_start, "end": audio_length})
        
        # 合并相邻或重叠的语音段
        merged_segments = self.merge_overlapping_segments(raw_voice_segments)
        
        # 对合并后的语音段进行扩展处理，并确保扩展后没有重叠
        voice_segments = []
        expand_samples = int(self.sample_rate * 0.3)
        
        # 先进行扩展
        for seg in merged_segments:
            expanded_start = max(0, seg["start"] - expand_samples)
            expanded_end = min(audio_length, seg["end"] + expand_samples)
            voice_segments.append({"start": expanded_start, "end": expanded_end})
        
        # 对扩展后的语音段再次合并，确保没有重叠
        voice_segments = self.merge_overlapping_segments(voice_segments)
        
        # 统计结果
        total_voice_duration = sum(seg["end"] - seg["start"] for seg in voice_segments) / self.sample_rate
        print(f"{len(voice_segments)} 个语音段，总时长: {total_voice_duration:.2f}秒")
        
        # 显示语音段信息
        if voice_segments:
            for i, seg in enumerate(voice_segments):
                start_time = seg["start"] / self.sample_rate
                end_time = seg["end"] / self.sample_rate
                duration = end_time - start_time
                print(f"语音段 {i+1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
        else:
            print("警告：未检测到任何语音段！")
        
        # 创建只有语音的音频
        voice_audio = np.zeros_like(audio_data)
        for seg in voice_segments:
            voice_audio[seg["start"]:seg["end"]] = audio_data[seg["start"]:seg["end"]]
        
        # 如果没有检测到语音，保存原始音频
        if len(voice_segments) == 0:
            print("将保存原始音频作为备用")
            voice_audio = audio_data.copy()
        
        return voice_audio, voice_segments
        
    def merge_overlapping_segments(self, segments):
        """合并重叠或相邻的语音段"""
        if not segments:
            return []
            
        # 按起始时间排序
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        merged = [sorted_segments[0].copy()]
        
        for current in sorted_segments[1:]:
            last = merged[-1]
            
            # 检查当前段与上一段是否重叠或比较接近（间隔小于0.2秒）
            time_gap = current["start"] - last["end"]
            
            if time_gap <= 0.2 * self.sample_rate:  # 如果重叠或间隔小于0.2秒
                # 合并段的起始时间取最小值，结束时间取最大值
                last["start"] = min(last["start"], current["start"])
                last["end"] = max(last["end"], current["end"])
            else:
                # 否则添加新段
                merged.append(current.copy())
        
        return merged
    
    def run_test(self):
        """运行测试"""
        # 使用jfk.wav文件
        audio_path = "jfk.wav"
        if not os.path.exists(audio_path):
            print(f"错误：找不到音频文件 {audio_path}")
            return
        
        # 预处理音频
        audio_data = self.preprocess_audio(audio_path)
        
        # 检测语音段
        voice_audio, voice_segments = self.detect_voice_segments(audio_data)
        
        # 保存结果
        sf.write("fixed_original.wav", audio_data, self.sample_rate)
        sf.write("fixed_vad_processed.wav", voice_audio, self.sample_rate)
        
        print("\n测试结果：")
        print("已生成 fixed_original.wav 和 fixed_vad_processed.wav")
        
        # 检查生成的VAD处理后音频是否有内容
        max_amp = np.max(np.abs(voice_audio))
        if max_amp > 0:
            print("✅ VAD处理后的音频包含声音信号")

if __name__ == "__main__":
    tester = FixedVADTest()
    tester.run_test()
    print("测试完成")