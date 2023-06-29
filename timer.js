const Timer = document.getElementById("Timer"); //스코어 기록창-분
let time = 8000;
let sec = 8;

Timer.value = "8";

function TIMER() {
  PlAYTIME = setInterval(function () {
    time = time - 1000; //1초씩 줄어듦
    min = time / (60 * 1000); //초를 분으로 나눠준다.

    if (sec > 0) {
      //sec=60 에서 1씩 빼서 출력해준다.
      sec = sec - 1;
      Timer.value = sec; //실수로 계산되기 때문에 소숫점 아래를 버리고 출력해준다.
    }
    if (sec === 0) {
      // 0에서 -1을 하면 -59가 출력된다.
      // 그래서 0이 되면 바로 sec을 60으로 돌려주고 value에는 0을 출력하도록 해준다.
      sec = 60;
      Timer.value = "Time Out!";
    }
  }, 1000); //1초마다
}

TIMER();
setTimeout(function () {
  clearInterval(PlAYTIME);
}, 8000); //3분이 되면 타이머를 삭제한다.
