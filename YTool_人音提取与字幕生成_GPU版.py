import numpy as np
import soundfile as sf
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont
import shutil
from faster_whisper import WhisperModel
import time
import subprocess
import tempfile
import re
import onnxruntime as ort

class AudioTranscriptionTool:
    def __init__(self, root):
        # 主窗口初始化
        self.root = root
        self.root.title("人音提取与字幕生成工具(GPU加速版)")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        
        # 设置中文字体 - 修复显示问题
        self.font_config()
        
        # 设置VAD参数
        self.vad_sample_rate = 16000
        self.window_size = 512  # 32ms @ 16kHz
        self.vad_model = None
        self.whisper_model = None
        self.audio_path = None
        self.original_sample_rate = None
        self.original_audio_data = None
        self.voice_segments = None
        self.output_folder = "AudioFiles"
        self.temp_folder = tempfile.gettempdir()
        self.enhanced_audio_path = None
        self.selected_subtitle_format = tk.StringVar(value="srt")
        self.supported_subtitle_formats = ["srt", "vtt", "ass", "txt"]
        
        # 文件路径显示变量
        self.file_path_var = tk.StringVar(value="未选择文件")
        
        # 检查FFmpeg是否可用
        self.ffmpeg_available = self._check_ffmpeg_availability()
        
        # 预设的音频增强参数
        self.noise_reduction_var = tk.BooleanVar(value=True)
        self.voice_enhancement_var = tk.BooleanVar(value=True)
        self.volume_normalization_var = tk.BooleanVar(value=True)
        self.eq_adjustment_var = tk.BooleanVar(value=True)
        
        # 进度条相关变量
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="就绪")
        
        # 创建GUI界面
        self.create_gui()
    
    def font_config(self):
        """配置中文字体 - 使用Segoe UI提高美观度"""
        # 使用Segoe UI作为主字体，在Windows系统中会自动使用微软雅黑显示中文
        chosen_font = "Segoe UI"
        
        # 配置默认字体
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family=chosen_font, size=10)
        self.root.option_add("*Font", default_font)
        
        # 为ttk组件设置字体
        style = ttk.Style()
        style.configure(".", font=(chosen_font, 10))
    
    def create_gui(self):
        """创建GUI界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题标签 - 突出显示GPU加速
        title_label = ttk.Label(main_frame, text="人音提取与字幕生成工具(GPU加速版)", 
                               font=("Segoe UI", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_path_label = ttk.Label(file_frame, textvariable=self.file_path_var, wraplength=400)
        file_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        select_file_btn = ttk.Button(file_frame, text="选择文件", command=self.select_file)
        select_file_btn.pack(side=tk.RIGHT)
        
        # 字幕格式选择区域
        format_frame = ttk.LabelFrame(main_frame, text="选择字幕格式", padding="10")
        format_frame.pack(fill=tk.X, pady=(0, 10))
        
        format_frame_inner = ttk.Frame(format_frame)
        format_frame_inner.pack(fill=tk.X)
        
        for fmt in self.supported_subtitle_formats:
            ttk.Radiobutton(format_frame_inner, text=fmt.upper(), variable=self.selected_subtitle_format, 
                           value=fmt).pack(side=tk.LEFT, padx=15)
        
        # 音频增强选项区域
        enhance_frame = ttk.LabelFrame(main_frame, text="音频增强选项", padding="10")
        enhance_frame.pack(fill=tk.X, pady=(0, 10))
        
        if self.ffmpeg_available:
            enhance_options_frame = ttk.Frame(enhance_frame)
            enhance_options_frame.pack(fill=tk.X)
            
            ttk.Checkbutton(enhance_options_frame, text="噪音消除", variable=self.noise_reduction_var).pack(anchor="w", pady=2)
            ttk.Checkbutton(enhance_options_frame, text="语音增强", variable=self.voice_enhancement_var).pack(anchor="w", pady=2)
            ttk.Checkbutton(enhance_options_frame, text="音量归一化", variable=self.volume_normalization_var).pack(anchor="w", pady=2)
            ttk.Checkbutton(enhance_options_frame, text="均衡器调整", variable=self.eq_adjustment_var).pack(anchor="w", pady=2)
        else:
            ttk.Label(enhance_frame, text="提示: FFmpeg未安装，无法使用音频增强功能", foreground="red").pack(fill=tk.X)
        
        # 状态和进度区域
        status_frame = ttk.LabelFrame(main_frame, text="处理状态", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(fill=tk.X, pady=(0, 5))
        
        progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.X)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        start_btn = ttk.Button(button_frame, text="开始处理", command=self.start_processing, style="Accent.TButton")
        start_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        exit_btn = ttk.Button(button_frame, text="退出", command=self.root.destroy)
        exit_btn.pack(side=tk.RIGHT)
        
        # 设置按钮样式
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
    
    def _check_ffmpeg_availability(self):
        """检查FFmpeg是否可用"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def select_file(self):
        """选择音视频文件"""
        # 支持的文件类型
        filetypes = [
            ("所有支持的媒体文件", "*.wav *.mp3 *.flac *.m4a *.ogg *.opus *.mp4 *.avi *.mov *.mkv *.wmv *.webm"),
            ("音频文件", "*.wav *.mp3 *.flac *.m4a *.ogg *.opus"),
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm"),
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择音视频文件",
            filetypes=filetypes
        )
        
        if not file_path:
            return
        
        self.audio_path = file_path
        self.file_path_var.set(file_path)
        
        # 对于非WAV文件或视频文件，使用FFmpeg提取音频
        _, ext = os.path.splitext(file_path.lower())
        if ext != '.wav' or self._is_video_file(file_path):
            if not self.ffmpeg_available:
                messagebox.showerror("错误", "FFmpeg未正确安装或未添加到PATH中")
                self.audio_path = None
                self.file_path_var.set("未选择文件")
                return
            
            # 显示加载状态
            self.status_var.set("正在提取音频...")
            self.root.update()
            
            # 提取音频到临时WAV文件
            temp_wav_path = os.path.join(self.temp_folder, f"temp_{int(time.time())}.wav")
            try:
                self._extract_audio_with_ffmpeg(file_path, temp_wav_path)
                # 读取提取的音频
                self.original_audio_data, self.original_sample_rate = sf.read(temp_wav_path, dtype='float32')
                # 删除临时文件
                os.remove(temp_wav_path)
                self.status_var.set(f"已加载文件，采样率: {self.original_sample_rate}Hz")
            except Exception as e:
                messagebox.showerror("错误", f"提取或读取音频失败: {str(e)}")
                self.audio_path = None
                self.file_path_var.set("未选择文件")
                self.status_var.set("就绪")
                return
        else:
            # 直接读取WAV文件
            try:
                self.original_audio_data, self.original_sample_rate = sf.read(file_path, dtype='float32')
                self.status_var.set(f"已加载文件，采样率: {self.original_sample_rate}Hz")
            except Exception as e:
                messagebox.showerror("错误", f"读取音频文件失败: {str(e)}")
                self.audio_path = None
                self.file_path_var.set("未选择文件")
                return
        
        # 转换为单声道
        if len(self.original_audio_data.shape) > 1:
            self.original_audio_data = np.mean(self.original_audio_data, axis=1)
    
    def _is_video_file(self, file_path):
        """检查文件是否为视频文件"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm']
        _, ext = os.path.splitext(file_path.lower())
        return ext in video_extensions
    
    def _extract_audio_with_ffmpeg(self, input_path, output_path):
        """使用FFmpeg从音视频文件中提取音频"""
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vn",  # 禁用视频
            "-ac", "1",  # 转换为单声道
            "-ar", "44100",  # 设置采样率
            "-f", "wav",  # 设置输出格式
            "-y",  # 覆盖现有文件
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg提取音频失败: {result.stderr}")
    
    def load_vad_model(self):
        """加载VAD模型 - 优化GPU加速配置"""
        try:
            vad_model_path = "silero-vad/onnx/model.onnx"
            if not os.path.exists(vad_model_path):
                raise FileNotFoundError(f"VAD模型文件不存在: {vad_model_path}")
            
            # 检查可用的提供程序
            available_providers = ort.get_available_providers()
            self.status_var.set(f"可用推理后端: {', '.join(available_providers)}")
            self.root.update()
            
            # 尝试配置ONNX Runtime使用GPU加速
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                # 使用CUDA加速
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                    'cuda_mem_limit': 2 * 1024 * 1024 * 1024  # 2GB GPU内存限制
                }))
            elif 'DmlExecutionProvider' in available_providers and os.name == 'nt':  # Windows上的DirectML
                providers.append('DmlExecutionProvider')
            # 始终添加CPU作为后备
            providers.append('CPUExecutionProvider')
            
            self.vad_model = ort.InferenceSession(vad_model_path, providers=providers)
            
            # 检查是否成功使用GPU
            if 'CUDAExecutionProvider' in providers:
                self.status_var.set("VAD模型已加载 (NVIDIA GPU加速)")
            elif 'DmlExecutionProvider' in providers:
                self.status_var.set("VAD模型已加载 (DirectML GPU加速)")
            else:
                self.status_var.set("VAD模型已加载 (CPU模式)")
            
            return True
        except Exception as e:
            print(f"加载VAD模型时出错: {str(e)}")
            # 如果GPU加载失败，尝试回退到CPU
            try:
                self.vad_model = ort.InferenceSession(vad_model_path, providers=['CPUExecutionProvider'])
                self.status_var.set("VAD模型已加载 (CPU模式)")
                return True
            except Exception as cpu_e:
                messagebox.showerror("错误", f"回退到CPU模式也失败: {str(cpu_e)}")
                return False
    
    def load_whisper_model(self):
        """加载Whisper模型 - 优化GPU加速配置"""
        try:
            model_path = "faster-whisper-medium"
            if not os.path.exists(model_path):
                # 如果本地模型不存在，让faster-whisper自动下载
                self.status_var.set("本地模型不存在，正在尝试自动下载模型...")
                self.root.update()
                model_path = "medium"
            
            # 尝试使用GPU运行
            try:
                self.whisper_model = WhisperModel(model_path, device="cuda", compute_type="float16")
                self.status_var.set("Whisper模型已加载 (NVIDIA GPU加速)")
                return True
            except Exception as gpu_error:
                print(f"GPU模式加载失败: {str(gpu_error)}")
                # 检查是否有其他GPU选项
                try:
                    # 尝试使用DirectML（Windows上可用）
                    import torch
                    if torch.cuda.is_available():
                        self.whisper_model = WhisperModel(model_path, device="cuda", compute_type="float32")
                        self.status_var.set("Whisper模型已加载 (CUDA兼容模式)")
                        return True
                    else:
                        # 检查是否有其他加速选项
                        if os.name == 'nt':  # Windows
                            # 尝试使用CPU但使用更高效的计算类型
                            self.whisper_model = WhisperModel(model_path, device="cpu", compute_type="int8")
                            self.status_var.set("Whisper模型已加载 (CPU优化模式)")
                            return True
                        else:
                            raise Exception("没有可用的GPU加速选项")
                except Exception:
                    # 如果所有GPU尝试都失败，回退到基础CPU模式
                    raise gpu_error
        except Exception as e:
            # 显示详细的错误信息，帮助用户排查问题
            error_msg = f"加载Whisper模型失败: {str(e)}\n\n请确保已正确安装并配置了:\n1. CUDA兼容的GPU驱动\n2. onnxruntime-gpu 1.22.0或更高版本\n3. faster-whisper库"
            print(error_msg)
            
            # 尝试最后一次使用CPU
            try:
                self.whisper_model = WhisperModel(model_path, device="cpu", compute_type="float32")
                self.status_var.set("Whisper模型已加载 (CPU模式)")
                return True
            except Exception as cpu_e:
                messagebox.showerror("错误", f"回退到CPU模式也失败: {str(cpu_e)}")
                return False
    
    def enhance_audio(self):
        """使用FFmpeg增强音频，使人声更明显"""
        if not self.ffmpeg_available:
            # 如果FFmpeg不可用，跳过增强步骤
            self.enhanced_audio_path = None
            return True
        
        if not any([self.noise_reduction_var.get(), self.voice_enhancement_var.get(), 
                    self.volume_normalization_var.get(), self.eq_adjustment_var.get()]):
            # 如果所有增强选项都未启用，跳过增强步骤
            self.enhanced_audio_path = None
            return True
        
        try:
            # 保存原始音频数据到临时文件，以便FFmpeg处理
            temp_input = os.path.join(self.temp_folder, f"temp_input_{int(time.time())}.wav")
            temp_output = os.path.join(self.temp_folder, f"temp_enhanced_{int(time.time())}.wav")
            
            sf.write(temp_input, self.original_audio_data, self.original_sample_rate)
            
            # 构建FFmpeg命令
            cmd = ["ffmpeg", "-i", temp_input]
            
            # 添加音频增强滤镜
            filters = []
            
            if self.noise_reduction_var.get():
                # 噪音消除滤镜
                filters.append("afftdn=nf=-25")
            
            if self.voice_enhancement_var.get():
                # 语音增强滤镜（使用压缩器和扩展器）
                filters.append("compand=0.02|0.02:0.1|0.1:-90/-60|-30/-10|0/-5:6:0:0:0")
            
            if self.eq_adjustment_var.get():
                # 均衡器调整（增强人声频率范围）
                filters.append("highpass=f=80, lowpass=f=5000, equalizer=f=1000:g=2, equalizer=f=250:g=1.5")
            
            if self.volume_normalization_var.get():
                # 音量归一化
                filters.append("loudnorm=I=-16:LRA=11:TP=-2")
            
            # 添加滤镜到命令
            if filters:
                filter_str = ",".join(filters)
                cmd.extend(["-af", filter_str])
            
            # 添加输出参数
            cmd.extend(["-y", temp_output])
            
            # 执行FFmpeg命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg音频增强失败: {result.stderr}")
            
            # 读取增强后的音频
            self.original_audio_data, self.original_sample_rate = sf.read(temp_output, dtype='float32')
            
            # 保存增强后的音频到输出文件夹
            self.enhanced_audio_path = os.path.join(self.output_folder, "enhanced_full_audio.wav")
            sf.write(self.enhanced_audio_path, self.original_audio_data, self.original_sample_rate)
            
            # 清理临时文件
            os.remove(temp_input)
            os.remove(temp_output)
            
            return True
        except Exception as e:
            print(f"音频增强过程中发生错误: {str(e)}")
            # 即使增强失败，也继续处理原始音频
            self.enhanced_audio_path = None
            return True
    
    def preprocess_audio_for_vad(self, audio_data, original_sr):
        """预处理音频用于VAD检测"""
        # 重采样到VAD所需的采样率
        if original_sr != self.vad_sample_rate:
            from resampy import resample
            audio_data = resample(audio_data, original_sr, self.vad_sample_rate)
        
        # 归一化
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        return audio_data
    
    def detect_voice_segments(self, audio_data):
        """检测语音段"""
        if self.vad_model is None:
            if not self.load_vad_model():
                return None
        
        # 状态形状应该是(2, 1, 128)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        
        # 分帧处理
        audio_length = len(audio_data)
        num_windows = int(np.ceil(audio_length / self.window_size))
        
        # 存储语音段（先不做扩展）
        raw_voice_segments = []
        in_voice = False
        segment_start = 0
        
        # 获取模型输入名称
        input_names = [input.name for input in self.vad_model.get_inputs()]
        
        for i in range(num_windows):
            # 更新进度
            if i % 100 == 0:
                progress = (i / num_windows) * 100
                self.progress_var.set(progress)
                self.root.update()
                
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
                ort_inputs['sr'] = np.array([self.vad_sample_rate], dtype=np.int64)
            
            # 推理
            try:
                ort_outs = self.vad_model.run(None, ort_inputs)
                vad_prob = float(ort_outs[0][0][0])
                
                # 更新状态
                if len(ort_outs) > 1:
                    state = ort_outs[1]
                
                # 检测语音段，使用较低的阈值以提高敏感度
                threshold = 0.15
                if vad_prob > threshold:
                    if not in_voice:
                        in_voice = True
                        segment_start = start
                else:
                    if in_voice:
                        in_voice = False
                        # 只添加有意义长度的语音段
                        if start - segment_start > self.vad_sample_rate * 0.1:  # 至少100ms
                            raw_voice_segments.append({"start": segment_start, "end": start})
            except Exception as e:
                pass  # 静默处理错误，继续下一个窗口
        
        # 处理最后一个语音段
        if in_voice and segment_start < audio_length:
            if audio_length - segment_start > self.vad_sample_rate * 0.1:
                raw_voice_segments.append({"start": segment_start, "end": audio_length})
        
        # 合并相邻或重叠的语音段
        merged_segments = self.merge_overlapping_segments(raw_voice_segments)
        
        # 对合并后的语音段进行扩展处理，并确保扩展后没有重叠
        voice_segments = []
        expand_samples = int(self.vad_sample_rate * 0.3)  # 每段前后扩展0.3秒
        
        # 先进行扩展
        for seg in merged_segments:
            expanded_start = max(0, seg["start"] - expand_samples)
            expanded_end = min(audio_length, seg["end"] + expand_samples)
            voice_segments.append({"start": expanded_start, "end": expanded_end})
        
        # 对扩展后的语音段再次合并，确保没有重叠
        voice_segments = self.merge_overlapping_segments(voice_segments)
        
        # 将VAD采样率下的时间转换回原始采样率下的时间
        ratio = self.original_sample_rate / self.vad_sample_rate
        for seg in voice_segments:
            seg["start"] = int(seg["start"] * ratio)
            seg["end"] = int(seg["end"] * ratio)
        
        return voice_segments
    
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
            
            if time_gap <= 0.2 * self.vad_sample_rate:  # 如果重叠或间隔小于0.2秒
                # 合并段的起始时间取最小值，结束时间取最大值
                last["start"] = min(last["start"], current["start"])
                last["end"] = max(last["end"], current["end"])
            else:
                # 否则添加新段
                merged.append(current.copy())
        
        return merged
    
    def clear_and_create_output_folder(self):
        """清空并创建输出文件夹"""
        try:
            # 如果文件夹存在，先清空
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)
            # 创建文件夹
            os.makedirs(self.output_folder)
            return True
        except Exception as e:
            messagebox.showerror("错误", f"创建输出文件夹失败: {str(e)}")
            return False
    
    def extract_voice_segments(self):
        """从原文件中提取语音段并保存"""
        if self.voice_segments is None or self.original_audio_data is None:
            messagebox.showerror("错误", "没有语音段数据或原始音频数据")
            return False
        
        if not self.clear_and_create_output_folder():
            return False
        
        try:
            segment_files = []
            for i, seg in enumerate(self.voice_segments):
                # 更新进度
                progress = 70 + (i / len(self.voice_segments)) * 10
                self.progress_var.set(progress)
                self.root.update()
                
                # 从原始音频中提取语音段
                segment_audio = self.original_audio_data[seg["start"]:seg["end"]]
                
                # 计算时间（秒）
                start_time = seg["start"] / self.original_sample_rate
                end_time = seg["end"] / self.original_sample_rate
                
                # 生成文件名，使用时间信息
                file_name = f"segment_{i+1}_{start_time:.2f}s_{end_time:.2f}s.wav"
                file_path = os.path.join(self.output_folder, file_name)
                
                # 保存语音段
                sf.write(file_path, segment_audio, self.original_sample_rate)
                segment_files.append({
                    "path": file_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "index": i+1
                })
            
            return segment_files
        except Exception as e:
            messagebox.showerror("错误", f"提取语音段失败: {str(e)}")
            return False
    
    def transcribe_audio_segments(self, segment_files):
        """转录语音段生成字幕"""
        if not segment_files:
            messagebox.showerror("错误", "没有语音段文件可供转录")
            return False
        
        if self.whisper_model is None:
            if not self.load_whisper_model():
                return False
        
        try:
            # 创建字幕文件
            subtitle_file_path = f"transcription.{self.selected_subtitle_format.get()}"
            total_segments = len(segment_files)
            
            with open(subtitle_file_path, "w", encoding="utf-8") as subtitle_file:
                for i, seg_info in enumerate(segment_files):
                    # 更新进度
                    progress = 80 + (i / total_segments) * 20
                    self.progress_var.set(progress)
                    self.status_var.set(f"正在转录语音段 {i+1}/{total_segments} (GPU加速)")
                    self.root.update()
                    
                    file_path = seg_info["path"]
                    start_time = seg_info["start_time"]
                    end_time = seg_info["end_time"]
                    index = seg_info["index"]
                    
                    # 使用Whisper模型转录
                    segments, info = self.whisper_model.transcribe(file_path, language="zh")
                    
                    # 根据选择的格式转换字幕
                    subtitle_content = ""
                    format_type = self.selected_subtitle_format.get()
                    if format_type == "srt":
                        subtitle_content = self.format_srt(segments, start_time, index)
                    elif format_type == "vtt":
                        subtitle_content = self.format_vtt(segments, start_time, index)
                    elif format_type == "ass":
                        subtitle_content = self.format_ass(segments, start_time, index)
                    elif format_type == "txt":
                        subtitle_content = self.format_txt(segments, start_time, index)
                    
                    subtitle_file.write(subtitle_content)
                    
                    # 确保数据写入文件
                    subtitle_file.flush()
            
            # 验证文件是否创建且不为空
            if os.path.exists(subtitle_file_path) and os.path.getsize(subtitle_file_path) > 0:
                return True
            else:
                raise Exception("字幕文件生成失败或为空")
        except Exception as e:
            messagebox.showerror("错误", f"转录语音段失败: {str(e)}")
            return False
    
    def format_srt(self, segments, base_time, base_index):
        """将转录结果格式化为SRT格式"""
        srt_content = ""
        
        for i, segment in enumerate(segments):
            # 计算实际时间（加上语音段的起始时间）
            start_time = base_time + segment.start
            end_time = base_time + segment.end
            
            # 格式化时间为SRT格式：00:00:00,000
            start_str = self.format_time(start_time, format_type="srt")
            end_str = self.format_time(end_time, format_type="srt")
            
            # 写入SRT内容 - 使用纯数字递增编号，符合SRT规范
            subtitle_number = base_index + i
            srt_content += f"{subtitle_number}\n"
            srt_content += f"{start_str} --> {end_str}\n"
            srt_content += f"{segment.text.strip()}\n\n"
        
        return srt_content
    
    def format_vtt(self, segments, base_time, base_index):
        """将转录结果格式化为WebVTT格式"""
        vtt_content = ""
        
        for i, segment in enumerate(segments):
            # 计算实际时间（加上语音段的起始时间）
            start_time = base_time + segment.start
            end_time = base_time + segment.end
            
            # 格式化时间为VTT格式：00:00:00.000
            start_str = self.format_time(start_time, format_type="vtt")
            end_str = self.format_time(end_time, format_type="vtt")
            
            # 写入VTT内容
            vtt_content += f"{base_index}_{i+1}\n"
            vtt_content += f"{start_str} --> {end_str}\n"
            vtt_content += f"{segment.text.strip()}\n\n"
        
        return vtt_content
    
    def format_ass(self, segments, base_time, base_index):
        """将转录结果格式化为ASS格式"""
        ass_content = ""
        
        # ASS格式需要头部信息，但我们在主函数中添加
        for i, segment in enumerate(segments):
            # 计算实际时间（加上语音段的起始时间）
            start_time = base_time + segment.start
            end_time = base_time + segment.end
            
            # 格式化时间为ASS格式：0:00:00.00
            start_str = self.format_time(start_time, format_type="ass")
            end_str = self.format_time(end_time, format_type="ass")
            
            # ASS格式行
            line = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{segment.text.strip()}\n"
            ass_content += line
        
        return ass_content
    
    def format_txt(self, segments, base_time, base_index):
        """将转录结果格式化为纯文本时间轴格式"""
        txt_content = ""
        
        for i, segment in enumerate(segments):
            # 计算实际时间（加上语音段的起始时间）
            start_time = base_time + segment.start
            end_time = base_time + segment.end
            
            # 格式化时间为文本格式：[00:00:00] - [00:00:00]
            start_str = self.format_time(start_time, format_type="txt")
            end_str = self.format_time(end_time, format_type="txt")
            
            # 写入文本内容
            txt_content += f"[{start_str}] - [{end_str}] {segment.text.strip()}\n"
        
        return txt_content
    
    def format_time(self, seconds, format_type="srt"):
        """将秒转换为不同的时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        
        if format_type == "srt":
            # SRT格式: 00:00:00,000
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        elif format_type == "vtt":
            # VTT格式: 00:00:00.000
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        elif format_type == "ass":
            # ASS格式: 0:00:00.00
            return f"{hours}:{minutes:02d}:{secs:02d}.{millis//10:02d}"
        elif format_type == "txt":
            # 文本格式: 00:00:00
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            # 默认返回SRT格式
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def start_processing(self):
        """开始处理流程"""
        if not self.audio_path:
            messagebox.showinfo("提示", "请先选择一个音视频文件")
            return
        
        try:
            # 重置进度条和状态
            self.progress_var.set(0)
            self.status_var.set("开始处理...")
            self.root.update()
            
            # 1. 加载VAD模型（进度10%）
            self.status_var.set("正在加载VAD模型...(GPU加速)")
            self.root.update()
            if not self.load_vad_model():
                self.status_var.set("加载模型失败")
                self.progress_var.set(0)
                return
            self.progress_var.set(10)
            self.root.update()
            
            # 2. 音频增强处理（进度20%-30%）
            self.status_var.set("正在进行音频增强处理...")
            self.root.update()
            if not self.enhance_audio():
                self.status_var.set("音频增强失败，继续使用原始音频")
                self.root.update()
                time.sleep(1)
            self.progress_var.set(30)
            self.root.update()
            
            # 3. 预处理音频用于VAD检测（进度40%）
            self.status_var.set("正在预处理音频...")
            self.root.update()
            vad_audio_data = self.preprocess_audio_for_vad(self.original_audio_data, self.original_sample_rate)
            self.progress_var.set(40)
            self.root.update()
            
            # 4. 检测语音段（进度40%-70%）
            self.status_var.set("正在检测语音段...")
            self.root.update()
            self.voice_segments = self.detect_voice_segments(vad_audio_data)
            if not self.voice_segments:
                messagebox.showinfo("提示", "未检测到语音段")
                self.status_var.set("未检测到语音段")
                self.progress_var.set(0)
                return
            self.progress_var.set(70)
            self.status_var.set(f"成功检测到 {len(self.voice_segments)} 个语音段")
            self.root.update()
            time.sleep(1)
            
            # 5. 提取语音段（进度70%-80%）
            self.status_var.set("正在提取语音段...")
            self.root.update()
            segment_files = self.extract_voice_segments()
            if not segment_files:
                self.status_var.set("提取语音段失败")
                self.progress_var.set(0)
                return
            self.progress_var.set(80)
            self.root.update()
            
            # 6. 加载Whisper模型并转录（进度80%-100%）
            self.status_var.set("正在加载Whisper模型...(GPU加速)")
            self.root.update()
            if self.transcribe_audio_segments(segment_files):
                # 处理完成
                self.progress_var.set(100)
                self.status_var.set("处理完成")
                
                # 创建完成消息弹窗
                self.show_completion_message()
            else:
                self.status_var.set("转录失败")
                self.progress_var.set(0)
        except Exception as e:
            messagebox.showerror("错误", f"程序运行失败: {str(e)}")
            self.status_var.set(f"运行错误: {str(e)}")
            self.progress_var.set(0)
    
    def show_completion_message(self):
        """显示完成消息"""
        # 创建一个新窗口显示完成消息
        completion_window = tk.Toplevel(self.root)
        completion_window.title("完成")
        completion_window.geometry("350x180")
        completion_window.resizable(False, False)
        
        # 确保窗口在最上层
        completion_window.attributes('-topmost', True)
        
        # 创建标签显示完成消息
        subtitle_file = f"transcription.{self.selected_subtitle_format.get()}"
        label_text = f"人音提取与字幕生成已完成！(GPU加速版)\n\n"
        label_text += f"字幕文件已保存为：{subtitle_file}\n"
        label_text += f"语音段已保存到：{self.output_folder}\n"
        if self.enhanced_audio_path:
            label_text += f"增强后的音频：enhanced_full_audio.wav"
        
        label = tk.Label(completion_window, text=label_text,
                        font=("Segoe UI", 10), wraplength=320, justify="center")
        label.pack(pady=20)
        
        # 创建确认按钮
        def on_ok():
            completion_window.destroy()
        
        button = tk.Button(completion_window, text="确定", command=on_ok, width=10)
        button.pack(pady=10)
        
        # 居中显示窗口
        completion_window.update_idletasks()
        width = completion_window.winfo_width()
        height = completion_window.winfo_height()
        x = (completion_window.winfo_screenwidth() // 2) - (width // 2)
        y = (completion_window.winfo_screenheight() // 2) - (height // 2)
        completion_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # 设置模态窗口
        completion_window.transient(self.root)
        completion_window.grab_set()
        
        # 运行消息框
        completion_window.wait_window()

def main():
    # 创建主窗口
    root = tk.Tk()
    
    # 创建应用程序实例
    app = AudioTranscriptionTool(root)
    
    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()