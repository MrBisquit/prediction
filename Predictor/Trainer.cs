using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using System.Diagnostics;

namespace Predictor
{
    public static class Trainer
    {
        public static void Train(string input,string output,int predictionDistance)
        {
            Stopwatch sw = Stopwatch.StartNew();

            var (X, Y) = ReadingContainer.BuildTensors(input, predictionDistance);

            Console.WriteLine("Tensors built, training...");

            var model = new WeatherNet();
            var opt = optim.Adam(model.parameters(), 0.01f);

            float oloss = 0;

            for (int i = 0; i < 10000; i++)
            {
                opt.zero_grad();

                var pred = model.forward(X);
                var loss = functional.huber_loss(pred, Y);

                loss.backward();
                opt.step();

                if (i % 10 == 0)
                    Console.WriteLine($"Epoch {i.ToString().PadLeft(5, '0')}, Loss: {loss.item<float>():F4}, Elapsed: {sw.Elapsed} " +
                        $"(Avg per 100: {(i >= 100 ? sw.Elapsed / (i / 100): "N/A")}), Complete: {Math.Round((float)((float)i / 10000f) * 100f, 2)}%");

                if (i % 1000 == 0)
                    model.save(output);

                oloss = loss.item<float>();
            }

            model.save(output);

            Console.WriteLine($"Training completed with loss: {oloss:F4}. Saved model to {output}");

            Utilizer.Predict(WeatherReading.TestReading,0,output);
        }
    }
}