using OpenCvSharp;
using OpenVinoSharp;
using System;

namespace OpenVINO
{
    public abstract class OnnxModel : IDisposable
    {
        public string model_path;

        public Core core; // 模型推理器
        public CompiledModel model;
        public InferRequest infer;

        public OnnxModel(string model_path, string device_name = "AUTO")
        {
            this.model_path = model_path;

            core = new Core();
            model = core.compile_model(model_path, device_name);
            infer = model.create_infer_request();
        }

        public abstract Result predict(Mat image);
        public abstract void draw(Result result, Mat image);

        public virtual void Dispose()
        {
            GC.SuppressFinalize(this);
        }

        ~OnnxModel()
        {
            Dispose();
        }
    }
}
