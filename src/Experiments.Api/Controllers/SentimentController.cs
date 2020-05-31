using System.ComponentModel.DataAnnotations;
using Experiments.Api.ML.SentimentAnalysis;
using Microsoft.AspNetCore.Mvc;

namespace Experiments.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class SentimentController : ControllerBase
    {
        private readonly SentimentAnalysisModel model;
        public SentimentController()
        {
            this.model = new SentimentAnalysisModel();
        }

        [HttpGet]
        [Route("predict")]
        public ActionResult<float> GetPrediction([FromQuery, Required] string text)
        {
            var observation = new SentimentObservation { Text = text };

            var prediction = this.model.Predict(observation);

            return prediction.Probability * 100;
        }

        [HttpPost]
        [Route("train")]
        public ActionResult PostTrain()
        {
            this.model.Train();

            var metrics = this.model.Evaluate();

            this.model.Save();

            return this.Ok(metrics);
        }
    }
}
