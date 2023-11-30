using Microsoft.Win32;
using OpenCvSharp;
using OpenVINO.Model;
using OpenVinoSharpPPTinyPose;
using System;
using System.IO;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;

namespace OpenVINO
{
    public partial class MainWindow : System.Windows.Window
    {
        public static Action<string> Logger, Message;

        private byte[] Image_Data = null;
        private string file_type = null;
        private VideoCapture video;
        private Thread thread, ImageThread;
        private bool busy = false, SpeedMode = false;

        public MainWindow()
        {
            InitializeComponent();

            Logger = m => Log(m);
            Message = m => ShowMoveMessageBox(m);
        }

        // 弹窗函数
        private void ShowMoveMessageBox(string Message)
        {
            Dispatcher.InvokeAsync(() =>
            {
                AnimationTimeline timeline = new DoubleAnimation()
                {
                    From = 0,
                    To = 1,
                    Duration = TimeSpan.FromSeconds(1)
                };

                try
                {
                    if (thread != null) thread.Abort();
                }
                catch { }

                thread = new Thread(() =>
                {
                    Thread.Sleep(2000);

                    Dispatcher.Invoke(() =>
                    {
                        AnimationTimeline timeline_2 = new DoubleAnimation()
                        {
                            From = 1,
                            To = 0,
                            Duration = TimeSpan.FromSeconds(1)
                        };

                        AnimationTimeline timeline_3 = new DoubleAnimation()
                        {
                            From = 0,
                            To = 100,
                            Duration = TimeSpan.FromSeconds(1)
                        };

                        MoveMessageBox.BeginAnimation(OpacityProperty, timeline_2);
                        MoveMessageBox.RenderTransform.BeginAnimation(TranslateTransform.YProperty, timeline_3);
                    });
                });
                thread.Start();

                (MoveMessageBox.Child as TextBlock).Text = Message;

                MoveMessageBox.BeginAnimation(OpacityProperty, timeline);
            });
        }

        // 日志输出
        private void Log(string message, SolidColorBrush solid = null, bool Trustmsg = false)
        {
            if (file_type == "video" && !Trustmsg) return;

            Dispatcher.InvokeAsync(() =>
            {
                Run textBlock = new Run()
                {
                    Text = message,
                    FontSize = 13,
                    FontWeight = FontWeights.Bold
                };

                if (solid != null)
                {
                    textBlock.Foreground = solid;
                }

                ConsoleBox.Document.Blocks.Add(new Paragraph(textBlock) { LineHeight = 1 });
            });
        }

        // Byte[]转Bitmap
        private BitmapImage ByteArrayToBitmapImage(byte[] byteArray)
        {
            BitmapImage bitmapImage = new BitmapImage();
            using (MemoryStream stream = new MemoryStream(byteArray))
            {
                stream.Position = 0;
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = stream;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
            }

            return bitmapImage;
        }

        // 打开图片按钮事件
        private void OpenImage(object sender, RoutedEventArgs e)
        {
            OpenFileDialog open = new OpenFileDialog();
            open.Filter = "图片(*.png;*.jpg)|*.png;*.jpg|视频(*.mp4)|*.mp4";

            open.ShowDialog();

            if (open.FileName != string.Empty)
            {
                if (ImageThread != null)
                {
                    ImageThread.Abort();
                    busy = false;
                }

                switch (Path.GetExtension(open.FileName))
                {
                    case ".png":
                    case ".jpg":
                        Image_Data = File.ReadAllBytes(open.FileName);
                        Beform.Source = ByteArrayToBitmapImage(Image_Data);

                        file_type = "image";

                        break;

                    case ".mp4":
                        video = VideoCapture.FromFile(open.FileName);
                        file_type = "video";

                        Mat output = new Mat();
                        video.Read(output);

                        Image_Data = output.ToBytes();
                        Beform.Source = ByteArrayToBitmapImage(Image_Data);

                        output.Release();

                        break;
                }
            }
        }

        // 开始识别按钮事件
        private void Launch(object sender, RoutedEventArgs e)
        {
            if (busy) return;

            ConsoleBox.Document.Blocks.Clear();
            After.Source = null;

            if (Image_Data == null || file_type == null)
            {
                ShowMoveMessageBox("图片是空的");
                return;
            }

            busy = true;

            ImageThread = new Thread(() =>
            {
                Log($"File_Type: {file_type}", Trustmsg: true);

                if (!Directory.Exists("./model"))
                {
                    Directory.CreateDirectory("./model");
                    ShowMoveMessageBox("未找到任何模型文件 请手动将模型放入程序根目录下model文件夹");

                    return;
                }

                try
                {
                    // 下面可以选择模型 添加到models即可直接使用

                    //PicoDet pico = new PicoDet("./model/picodet_s_320_lcnet_pedestrian.onnx", "CPU");   
                    PPTinyPose tinypose = new PPTinyPose("./model/yolov8s-pose.onnx", "CPU");
                    YolovDet yolov = new YolovDet("./model/det_lable.txt", "./Model/yolov8s.onnx", "CPU");

                    OnnxModel[] models = { tinypose, yolov };

                    switch (file_type)
                    {
                        // 图片
                        case "image":
                            UseOpenVINO(models);
                            break;

                        // 视频
                        case "video":
                            long tick = DateTime.Now.Ticks;
                            double fps = 0;
                            int count = 0;

                            while (video.IsOpened())
                            {
                                Mat output = new Mat();
                                video.Read(output);

                                count += 1;

                                if (!SpeedMode) Log($"Frame Count: {count} / {video.FrameCount}", Trustmsg: true);

                                // 将FPS打印到图片上
                                Cv2.PutText(output, fps.ToString("#0.00"), new OpenCvSharp.Point(5, 50), HersheyFonts.Italic, 1.0, new Scalar(0, 0, 255));

                                Image_Data = output.ToBytes();
                                Dispatcher.Invoke(() => Beform.Source = ByteArrayToBitmapImage(Image_Data));

                                UseOpenVINO(models);

                                // 计算FPS
                                double interval = TimeSpan.FromTicks(DateTime.Now.Ticks - tick).Milliseconds / 1000.0; // 1帧要多少秒
                                fps = 1.0 / interval;
                                tick = DateTime.Now.Ticks;

                                output.Release();
                            }

                            break;
                    }
                }
                catch (InvalidOperationException)
                {
                    Log($"发生了错误", new SolidColorBrush(Colors.Red));
                }
                catch (ThreadAbortException) 
                {
                    
                }
                catch (Exception ex)
                {
                    ShowMoveMessageBox(ex.ToString());
                }

                Log($"\n运行完成", Trustmsg: true);

                busy = false;
            });

            ImageThread.Start();
        }

        // 调用yolov8函数
        private void UseOpenVINO(OnnxModel[] models)
        {
            Mat image = Mat.FromImageData(Image_Data);

            long tick = DateTime.Now.Ticks;

            foreach (OnnxModel model in models)
            {
                Result result = model.predict(image);
                model.draw(result, image);
            }

            if (!SpeedMode) Log($"图片预测耗时: {TimeSpan.FromTicks(DateTime.Now.Ticks - tick).Milliseconds}ms", Trustmsg:true);

            Dispatcher.InvokeAsync(() => After.Source = ByteArrayToBitmapImage(image.ImEncode()));
        }

        // 拖动事件
        private void Window_OnDrag(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left) {
                this.DragMove();
            }
        }

        // 改变窗口大小事件
        private void Window_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            double width = e.NewSize.Width;
            double height = e.NewSize.Height;
            width -= 280;

            After.Width = width / 2;
            Beform.Width = width / 2;

            ConsoleBox.Height = height - 150;
        }

        // 右上角按钮图片更改
        private void Window_StateChanged(object sender, EventArgs e)
        {
            if (WindowState == WindowState.Maximized) {
                MaximizeButton.Content = Geometry.Parse(Application.Current.Resources["Restore"].ToString());
            }
            else {
                MaximizeButton.Content = Geometry.Parse(Application.Current.Resources["Maximize"].ToString());
            }
        }

        // 窗口关闭事件
        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (ImageThread != null) ImageThread.Abort();
            if (thread != null) thread.Abort();
            if (video != null) video.Dispose();
        }

        // 顶部按钮事件
        private void TitleBarClick(object sender, RoutedEventArgs e)
        {
            if (sender is Button button)
            {
                switch (button.CommandParameter)
                {
                    case "Close":
                        Application.Current.Shutdown();
                        break;

                    case "Minimize":
                        this.WindowState = WindowState.Minimized;
                        break;

                    case "Maximize":
                        if (WindowState == WindowState.Maximized)
                        {
                            this.WindowState = WindowState.Normal;
                        }
                        else
                        {
                            this.WindowState = WindowState.Maximized;
                        }

                        break;
                }
            }
            else if (sender is ToggleButton toggle)
            {
                switch (toggle.CommandParameter)
                {
                    case "Speed":
                        SpeedMode = toggle.IsChecked.Value;
                        if (SpeedMode) {
                            ConsoleBox.Document.Blocks.Clear();
                        }

                        break;
                }
            }
        }
    }
}
