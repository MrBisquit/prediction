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
        private readonly Module<Tensor, Tensor> model;

        public WeatherNet() : base("WeatherNet")
        {
            model = Sequential(
                Linear(9, 64),
                GELU(),
                LayerNorm(64),

                Linear(64, 64),
                GELU(),
                LayerNorm(64),

                Linear(64, 64),
                GELU(),
                LayerNorm(64),

                Dropout(0.0001),

                Linear(64, 32),
                GELU(),

                Linear(32, 4)
            );

            RegisterComponents();
        }

        public override Tensor forward(Tensor x) => model.forward(x);
    }
}
