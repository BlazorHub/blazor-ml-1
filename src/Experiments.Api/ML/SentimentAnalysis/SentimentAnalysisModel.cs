using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace Experiments.Api.ML.SentimentAnalysis
{
    public class SentimentAnalysisModel
    {
        private const string MODEL_PATH = "ML//SentimentAnalysis//SentimentAnalysisModel.zip";
        private readonly MLContext mlContext;
        private TrainTestData trainTestData;
        private ITransformer model;
        public SentimentAnalysisModel()
        {
            this.mlContext = new MLContext();

            this.model = this.mlContext.Model.Load(MODEL_PATH, out var _);
        }

        public CalibratedBinaryClassificationMetrics Train()
        {
            this.trainTestData = this.LoadData();

            this.model = this.BuildAndTrainModel();

            return Evaluate();
        }

        public SentimentPrediction Predict(SentimentObservation observation)
        {
            var predictionFunction = this.mlContext.Model.CreatePredictionEngine<SentimentObservation, SentimentPrediction>(this.model);

            return predictionFunction.Predict(observation);
        }

        public void Save()
        {
            this.mlContext.Model.Save(this.model, trainTestData.TrainSet.Schema, MODEL_PATH);
        }

        private CalibratedBinaryClassificationMetrics Evaluate()
        {
            var predictions = this.model.Transform(this.trainTestData.TestSet);

            return this.mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");
        }

        private ITransformer BuildAndTrainModel()
        {
            var model = this.mlContext.Transforms.Text.FeaturizeText("Features", inputColumnName: nameof(SentimentObservation.Text))
                .Append(this.mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            return model.Fit(this.trainTestData.TrainSet);
        }

        private TrainTestData LoadData()
        {
            var dataPath = Path.Combine(Environment.CurrentDirectory, "ML", "SentimentAnalysis", "Data", "yelp_labelled.txt");

            var dataView = this.mlContext.Data.LoadFromTextFile<SentimentObservation>(dataPath, hasHeader: false);

            return this.mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
        }
    }
}
