const disconnect = document.querySelector(".disconnect");
const webCam = document.querySelector("#webcam");
const videoContainer = document.querySelector(".video-container");

navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    const video = document.getElementById("webcam");
    video.srcObject = stream;
  })
  .catch((error) => {
    console.error("웹캠 에러:", error);
  });

function handleDisconnect() {
  webCam.classList.toggle("hide");
  videoContainer.classList.toggle("video-container-red");
}

disconnect.addEventListener("click", handleDisconnect);
