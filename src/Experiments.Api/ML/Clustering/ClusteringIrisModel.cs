using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Experiments.Api.ML.Clustering
{
    public class ClusteringIrisModel
    {
        private const string MODEL_PATH = "ML//Clustering//ClusteringIrisModel.zip";
        private readonly MLContext mlContext;
        private IDataView data;
        private ITransformer model;
        public ClusteringIrisModel()
        {
            this.mlContext = new MLContext();

            this.model = this.mlContext.Model.Load(MODEL_PATH, out var _);
        }

        public ClusteringMetrics Train()
        {
            this.data = this.LoadData();

            this.model = this.BuildAndTrainModel();

            return Evaluate();
        }

        public IrisPrediction Predict(IrisObservation observation)
        {
            var predictionFunction = this.mlContext.Model.CreatePredictionEngine<IrisObservation, IrisPrediction>(this.model);

            return predictionFunction.Predict(observation);
        }

        public void Save()
        {
            this.mlContext.Model.Save(this.model, data.Schema, MODEL_PATH);
        }

        private ClusteringMetrics Evaluate()
        {
            var predictions = this.model.Transform(this.data);

            return this.mlContext.Clustering.Evaluate(predictions);
        }

        private ITransformer BuildAndTrainModel()
        {
            var model = this.mlContext.Transforms.Concatenate("Features", nameof(IrisObservation.SepalLength), nameof(IrisObservation.SepalWidth), nameof(IrisObservation.PetalLength), nameof(IrisObservation.PetalWidth))
                .Append(this.mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3));

            return model.Fit(this.data);
        }

        private IDataView LoadData()
        {
            var dataPath = Path.Combine(Environment.CurrentDirectory, "ML", "Clustering", "Data", "iris.data.csv");

            return this.mlContext.Data.LoadFromTextFile<IrisObservation>(dataPath, separatorChar: ',', hasHeader: false);
        }
    }
}
