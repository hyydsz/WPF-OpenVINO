using OpenCvSharp;
using System;

namespace OpenVINO.Results
{
    public class ImageToData
    {
        public static float[] ImageToDataWithNormal(Mat image, Size output_size, ulong size)
        {
            Cv2.Resize(image, image, output_size);

            double[] std_values = new double[] { 1.0 * 255, 1.0 * 255, 1.0 * 255 };

            Cv2.Split(image, out Mat[] rgb_channels); // 分离图片数据通道
            for (int i = 0; i < rgb_channels.Length; i++)
            {
                // 分通道依此对每一个通道数据进行归一化处理
                rgb_channels[i].ConvertTo(rgb_channels[i], MatType.CV_32FC1, 1.0 / std_values[i], 0.0);
            }
            Cv2.Merge(rgb_channels, image); // 合并图片数据通道

            return ImageWriteToFloats(image, size);
        }

        public static float[] ImageToDataWithNormalAndMean(Mat image, Size output_size, ulong size)
        {
            Cv2.Resize(image, image, output_size);

            double[] mean_values = new double[] { 1.0 * 255, 1.0 * 255, 1.0 * 255 };
            double[] std_values = new double[] { 0.229 * 255, 0.224 * 255, 0.225 * 255 };

            Cv2.Split(image, out Mat[] rgb_channels); // 分离图片数据通道
            for (int i = 0; i < rgb_channels.Length; i++)
            {
                // 分通道依此对每一个通道数据进行归一化处理
                rgb_channels[i].ConvertTo(rgb_channels[i], MatType.CV_32FC1, 1.0 / mean_values[i], (0.0 - mean_values[i]) / std_values[i]);
            }
            Cv2.Merge(rgb_channels, image); // 合并图片数据通道

            return ImageWriteToFloats(image, size);
        }

        public static float[] ImageToDataWithoutNormal(Mat image, Size output_size, ulong size)
        {
            Cv2.Resize(image, image, output_size);

            Cv2.Split(image, out Mat[] rgb_channels); // 分离图片数据通道
            for (int i = 0; i < rgb_channels.Length; i++)
            {
                // 分通道依此对每一个通道数据进行归一化处理
                rgb_channels[i].ConvertTo(rgb_channels[i], MatType.CV_32FC1, 1.0 / 1.0, 0.0);
            }
            Cv2.Merge(rgb_channels, image); // 合并图片数据通道

            return ImageWriteToFloats(image, size);
        }

        public static float[] ImageToDataWithAffineAndNormal(Mat image, Size output_size, ulong size)
        {
            Point center = new Point(image.Cols / 2, image.Rows / 2); // 变换中心
            Size input_size = new Size(image.Cols, image.Rows); // 输入尺寸
            int rot = 0; // 角度

            // 获取仿射变换矩阵
            Mat warp_mat = get_affine_transform(center, input_size, rot, output_size);
            // 仿射变化
            Cv2.WarpAffine(image, image, warp_mat, output_size);

            Cv2.Split(image, out Mat[] rgb_channels); // 分离图片数据通道
            for (int i = 0; i < rgb_channels.Length; i++)
            {
                // 分通道依此对每一个通道数据进行归一化处理
                rgb_channels[i].ConvertTo(rgb_channels[i], MatType.CV_32FC1, 1.0 / 255.0, 0.0);
            }
            Cv2.Merge(rgb_channels, image); // 合并图片数据通道

            return ImageWriteToFloats(image, size);
        }

        public static float[] ImageWriteToFloats(Mat image, ulong size)
        {
            float[] datas = new float[size];
            for (int h = 0; h < image.Height; h++)
            {
                for (int w = 0; w < image.Width; w++)
                {
                    for (int c = 0; c < image.Channels(); c++)
                    {
                        datas[c * image.Width * image.Height + h * image.Width + w] = image.At<Vec3f>(h, w)[c];
                    }
                }
            }

            return datas;
        }

        public static Mat get_affine_transform(Point center, Size input_size, int rot, Size output_size, bool inv = false)
        {
            Point2f shift = new Point2f(0.0f, 0.0f);
            // 输入尺寸宽度
            int src_w = input_size.Width;

            // 输出尺寸
            int dst_w = output_size.Width;
            int dst_h = output_size.Height;

            // 旋转角度
            double rot_rad = Math.PI * rot / 180.0;
            double sn = Math.Sin(rot_rad);
            double cs = Math.Cos(rot_rad);

            float pt = src_w * -0.5F;

            Point2f src_dir = new Point2f((float)(-1.0F * pt * sn), (float)(pt * cs));
            Point2f dst_dir = new Point2f(0.0f, dst_w * -0.5F);

            Point2f[] src = new Point2f[3];

            src[0] = new Point2f(center.X + input_size.Width * shift.X, center.Y + input_size.Height * shift.Y);
            src[1] = new Point2f(center.X + src_dir.X + input_size.Width * shift.X, center.Y + src_dir.Y + input_size.Height * shift.Y);

            Point2f direction = src[0] - src[1];

            src[2] = new Point2f(src[1].X - direction.Y, src[1].Y - direction.X);

            Point2f[] dst = new Point2f[3];

            dst[0] = new Point2f(dst_w * 0.5F, dst_h * 0.5F);
            dst[1] = new Point2f(dst_w * 0.5F + dst_dir.X, dst_h * 0.5F + dst_dir.Y);

            direction = dst[0] - dst[1];

            dst[2] = new Point2f(dst[1].X - direction.Y, dst[1].Y - direction.X);

            // 是否为反向
            if (inv) { return Cv2.GetAffineTransform(dst, src); }
            else { return Cv2.GetAffineTransform(src, dst); }
        }

    }
}
