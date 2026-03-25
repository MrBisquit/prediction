using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace Predictor
{
    public class WeatherNet : Module<Tensor, Tensor>
    {
        private readonly Linear fc1, fc2, fc3;

        public WeatherNet() : base("WeatherNet")
        {
            fc1 = Linear(9, 256);
            fc2 = Linear(256, 256);
            fc3 = Linear(256, 4);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // this is pretty much a placeholder...
            x = functional.leaky_relu(fc1.forward(x));
            x = functional.gelu(fc2.forward(x));
            x = functional.gelu(fc3.forward(x));
            return x;
        }
    }
}
