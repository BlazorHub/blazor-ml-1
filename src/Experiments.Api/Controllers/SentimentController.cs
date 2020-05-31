using System;
using System.ComponentModel.DataAnnotations;
using System.IO;
using Experiments.Api.ML.SentimentAnalysis;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;

namespace Experiments.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class SentimentController : ControllerBase
    {
        private readonly PredictionEnginePool<SentimentObservation, SentimentPrediction> predictionEnginePool;
        public SentimentController(PredictionEnginePool<SentimentObservation, SentimentPrediction> predictionEnginePool)
        {
            this.predictionEnginePool = predictionEnginePool;
        }

        [HttpGet]
        [Route("predict")]
        public ActionResult<float> GetPrediction([FromQuery, Required] string text)
        {
            var sampleData = new SentimentObservation { Text = text };

            var prediction = this.predictionEnginePool.Predict(sampleData);

            float percentage = CalculatePercentage(prediction.Score);

            return percentage;
        }

        public float CalculatePercentage(double value)
        {
            return 100 * (1.0f / (1.0f + (float)Math.Exp(-value)));
        }

        [HttpPost]
        [Route("train")]
        public ActionResult PostTrain()
        {
            new SentimentAnalysisModel();

            return this.Ok();
        }
    }
}
