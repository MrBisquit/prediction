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
    public static class Trainer
    {
        public static void Train(string input,string output,int predictionDistance)
        {
            var (X, Y) = ReadingContainer.BuildTensors(input, predictionDistance);

            Console.WriteLine("Tensors built, training...");

            var model = new WeatherNet();
            var opt = optim.Adam(model.parameters(), 0.01f);

            for (int i = 0; i < 10000; i++)
            {
                opt.zero_grad();

                var pred = model.forward(X);
                var loss = functional.huber_loss(pred, Y);

                loss.backward();
                opt.step();

                if (i % 100 == 0)
                    Console.WriteLine($"Epoch {i}, Loss: {loss.item<float>():F4}");
            }

            model.save(output);

            Console.WriteLine($"Training completed. Saved model to {output}");

            Utilizer.Predict(WeatherReading.TestReading,0,output);
        }
    }
}
