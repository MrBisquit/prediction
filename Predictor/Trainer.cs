using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace Predictor
{
    public static class Trainer
    {
        public static void Train(string input, string output, int predictionDistance, int numepoch)
        {
            var (X, Y) = ReadingContainer.BuildTensors(input, predictionDistance);
            Console.WriteLine("Tensors built, training...");

            var device = cuda.is_available() ? CUDA : CPU;
            Console.WriteLine($"Using device: {device}");

            X = X.to(device);
            Y = Y.to(device);

            var model = new WeatherNet().to(device);
            var opt = optim.AdamW(model.parameters(), lr: 1e-3, weight_decay: 1e-4);
            var scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, numepoch);

            int batchSize = 256;
            int n = (int)X.shape[0];
            int numBatches = (n + batchSize - 1) / batchSize;

            float lastLoss = 0f;
            float bestLoss = float.MaxValue;

            for (int epoch = 0; epoch < numepoch; epoch++)
            {
                model.train();

                var indices = randperm(n, device: device);
                X = X.index_select(0, indices);
                Y = Y.index_select(0, indices);

                float epochLoss = 0f;

                for (int b = 0; b < numBatches; b++)
                {
                    int start = b * batchSize;
                    int end = Math.Min(start + batchSize, n);

                    using var xBatch = X.narrow(0, start, end - start);
                    using var yBatch = Y.narrow(0, start, end - start);

                    opt.zero_grad();

                    using var pred = model.forward(xBatch);
                    using var loss = functional.huber_loss(pred, yBatch);

                    loss.backward();

                    nn.utils.clip_grad_norm_(model.parameters(), 1.0);

                    opt.step();
                    epochLoss += loss.item<float>();
                }

                scheduler.step();

                epochLoss /= numBatches;
                lastLoss = epochLoss;

                if (epochLoss < bestLoss)
                {
                    bestLoss = epochLoss;
                    model.save(output + ".best");
                }

                if (epoch % 100 == 0)
                {
                    double currentLr = opt.ParamGroups.First().LearningRate;
                    Console.WriteLine($"Epoch {epoch}/{numepoch} | Loss: {epochLoss:F4} | Best: {bestLoss:F4} | LR: {currentLr:E2}");
                }
            }

            model.save(output);
            Console.WriteLine($"Training completed. Final loss: {lastLoss:F4}, Best loss: {bestLoss:F4}");
            Console.WriteLine($"Saved final model to {output}, best checkpoint to {output}.best");

            Utilizer.Predict(WeatherReading.TestReading, 0, output);
        }
    }
}
