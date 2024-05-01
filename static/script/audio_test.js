let start_button = document.querySelector("#start-record");
const timer = document.getElementById('timer');
const question = document.getElementById('question');

let timer_duration = 10 * 60 * 1000; // 10 minutes in milliseconds
let questions = ['QUESTIONS'];
let timeLeft = 600;
let timerInterval;

let question_number = 0;

function startRecording() {
  navigator.mediaDevices.getUserMedia({ 
    audio: {
      volume: 1.0 // set the audio volume to maximum
    }
  })
    .then(stream => {
      recorder = new MediaRecorder(stream);
      recorder.addEventListener('dataavailable', event => {
        chunks.push(event.data);
      });
      recorder.start();
    })
    .catch(error => {
      console.error(error);
    });
}

function startTimer() {
  timerInterval = setInterval(() => {
    timeLeft--;
    timer.style.width = `${(timeLeft / 600) * 100}%`;
    if (timeLeft <= 0) 
    {
      window.location.href = "/Audio_Test_Results";
    }
  }, 1000);
}

fetch('Questions')
  .then(response => response.json())
  .then(data => 
  {
    questions = questions.concat(data);
    showQuestion(0);
  });

async function showQuestion(index) 
{
  if (index < questions.length) 
  {
    question.textContent = questions[index];
    currentQuestionIndex = index;
  } else {
    question.textContent = "Processing"
    await new Promise(r => setTimeout(r, 10000));
    window.location.href = "/audio_dash";
  }
}

start_button.addEventListener('click', async function() {
  startTimer();
  showQuestion(1);
  startRecording();
});
