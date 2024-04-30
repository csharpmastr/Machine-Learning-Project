let acceptBtn = document.querySelector('#acceptModel');
var lastTime = 0;

// get video id
var videoId = document.getElementById('modelVideo');

// Function to navigate based on the previous page selection after accepting Model Limitations
function acceptModel() {
    // Get the selection from the URL parameters
    const params = new URLSearchParams(window.location.search);
    const previousPage = params.get('selection');
    // Redirect based on the previous page selection
    if (previousPage === 'entry') {
      window.location.href = "manualEntry.html";
    } else if (previousPage === 'upload') {
      window.location.href = "uploadImage.html";
    }
  }

function declineModel(){
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