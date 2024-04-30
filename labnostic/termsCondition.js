let acceptBtn = document.querySelector('#acceptVideo');
var lastTime = 0;

// get video id
var videoId = document.getElementById('termsVideo');

function acceptTerms(){
    window.location.href = "modelLimitations.html";
}

function declineTerms(){
    window.location.href = "homelab.html";
}


document.addEventListener("DOMContentLoaded", function(){

        // attach event listeners
        videoId.addEventListener('play', disableBtn());

        function disableBtn() {
            // if(videoId.onended)
            acceptBtn.disabled = true;
        }
        
        videoId.onplaying = function decreaseOpacity() {
            acceptBtn.disabled = true;
            acceptBtn.style.opacity = 0.5;
        }

        videoId.onended = function enableBtn() {
            // if(videoId.onended)
            acceptBtn.disabled = false;
            acceptBtn.style.opacity = 0.9;
        }

});