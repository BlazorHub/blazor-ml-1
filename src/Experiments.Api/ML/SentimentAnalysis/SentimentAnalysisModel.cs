using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace Experiments.Api.ML.SentimentAnalysis
{
    public class SentimentAnalysisModel
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "ML", "SentimentAnalysis", "Data", "yelp_labelled.txt");
        public ITransformer Model { get; set; }
        public SentimentAnalysisModel()
        {
            MLContext mlContext = new MLContext();

            TrainTestData trainTestData = this.LoadData(mlContext);

            this.Model = this.BuildAndTrainModel(mlContext, trainTestData.TrainSet);

            mlContext.Model.Save(this.Model, trainTestData.TrainSet.Schema, "ML//SentimentAnalysis//SentimentAnalysisModel.zip");
        }

        private ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentObservation.Text))
                                     .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            var model = estimator.Fit(trainSet);

            return model;
        }

        private TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentObservation>(_dataPath, hasHeader: false);

            return mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
        }

    }
}
