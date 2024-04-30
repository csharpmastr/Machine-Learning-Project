const firebaseConfig = {
    apiKey: "AIzaSyDGpBA5BOsWmOzGpkBp-tn9cTjzYZyXy_o",
    authDomain: "labnosticimages.firebaseapp.com",
    projectId: "labnosticimages",
    storageBucket: "labnosticimages.appspot.com",
    messagingSenderId: "459372180104",
    appId: "1:459372180104:web:8ae7050e15a2e4714c653d",
    measurementId: "G-LZRJKT61GE"
  };
  
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);

  // Selecting necessary DOM elements
  var fileText = document.querySelector(".chosenImage"); // Element to display selected file name
  var uploadPercentage = document.querySelector(".percentage"); // Element to display upload percentage
  var progress = document.querySelector(".progress"); // Progress bar
  var percentVal; // Variable to store upload percentage
  var fileItem; // Variable to store selected file
  var fileName; // Variable to store selected file name

  // Function to handle file selection
function getFile(e){
    fileItem = e.target.files[0]; // Get the selected file
    fileName = fileItem.name; // Get the file name
    fileText.innerHTML = fileName; // Display the file name

    // Check if a file is selected
    if (!fileItem) {
        return;
    }
  
    // Check if it's an image file
    if (!fileItem.type.startsWith("image/")) {
        alert("Please select an image file.");
        return;
    }

  }
  
  // Function to upload the selected image
  function uploadImage(){
    if (!fileItem || !fileItem.type.startsWith('image/')) {
        alert("Please select an image file."); // Display an error message if the selected file is not an image
        return;
      }
      
    let storageRef = firebase.storage().ref("images/"+fileName); // Reference to the storage location
    let uploadTask = storageRef.put(fileItem); // Upload task

    // Update progress as the upload progresses
    uploadTask.on("state_changed",(snapshot) =>{
      console.log(snapshot);
      percentVal = Math.floor((snapshot.bytesTransferred/snapshot.totalBytes)*100);
      console.log(percentVal);
      uploadPercentage.innerHTML = percentVal+"%"; // Display upload percentage
      progress.style.width = percentVal+"%"; // Update progress bar width

      // Enable/disable Analyze button based on upload progress
      if (percentVal === 100) {
        document.getElementById("analyze").disabled = false;
    } else {
        document.getElementById("analyze").disabled = true;
    }
    
    },(error)=>{
      console.log("Error is ", error); // Handle errors
    },()=>{
      // When upload is complete
      uploadTask.snapshot.ref.getDownloadURL().then((url)=>{
        console.log("URL", url);
      });
    });
  }

  // Event listener for Analyze button
    document.getElementById("analyze").addEventListener("click", function() {
    
    // Convert file object to data URL
    const reader = new FileReader();
    reader.onload = function(event) {
        const imageUrl = event.target.result;
        
        // Encode the image data as a URL parameter
        const encodedImageUrl = encodeURIComponent(imageUrl);
        
        // Redirect to image.html with the encoded image URL as a parameter
        window.location.href = `image.html?imageUrl=${encodedImageUrl}`;
    };
    reader.readAsDataURL(fileItem);
});