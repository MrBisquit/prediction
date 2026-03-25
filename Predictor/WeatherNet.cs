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
    public class ResidualBlock : Module<Tensor, Tensor>
    {
        private readonly Linear fc1, fc2;
        private readonly LayerNorm norm;
        private readonly Dropout drop;

        public ResidualBlock(long dim, double dropRate = 0.2) : base("ResidualBlock")
        {
            fc1 = Linear(dim, dim);
            fc2 = Linear(dim, dim);
            norm = LayerNorm(dim);
            drop = Dropout(dropRate);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var h = functional.gelu(fc1.forward(x));
            h = drop.forward(fc2.forward(h));
            return norm.forward(x + h);
        }
    }

    public class WeatherNet : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> model;

        public WeatherNet() : base("WeatherNet")
        {
            model = Sequential(
                Linear(9, 64),
                GELU(),
                LayerNorm(64),
                new ResidualBlock(64),
                new ResidualBlock(64),
                Linear(64, 32),
                GELU(),
                Linear(32, 4)
            );
            RegisterComponents();
        }

        public override Tensor forward(Tensor x) => model.forward(x);
    }
}
