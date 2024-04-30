// import {GlobalStorage} from './globalStorage.js';

// const global = GlobalStorage();

// document.addEventListener("DOMContentLoaded", function(){
//     // document.getElementById("entry").addEventListener("click", winLow("entry") );

//     // document.getElementById("upload").addEventListener("click", winLow("upload") );

//     // function winLow(source){
//     //     if (source === "entry") {
//     //         window.location.href = "termsCondition.html?source=manualEntry";
//     //     }
//     //     else if (source === "upload") {
//     //         window.location.href = "termsCondition.html?source=uploadImage";
//     //     }

//     // }
//     document.getElementById("entry").addEventListener("click", setvalue("entry"));
//     document.getElementById("upload").addEventListener("click", setvalue("upload"));

//         function setvalue(source){
//             if (source === "entry"){
//                 global.set("origin","entry");
//             }
//             else if (source === "upload"){
//                 global.set("origin","upload");
//             }
//         }

// });

function goTerms(selection) {
    window.location.href = "termsCondition.html?selection=" + selection;
  }