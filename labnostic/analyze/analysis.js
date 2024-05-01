function updateProgressCircleColor(value) {
    var progressCircle = document.querySelector('.progress-circle');
    var progressRingCircle = document.querySelector('.progress-ring-circle');
    
    if (value > 40 && value <= 60) {
        progressCircle.setAttribute('data-value', 'yellow');
    } else if (value > 60 && value <= 80) {
        progressCircle.setAttribute('data-value', 'orange');
    } else if (value > 80) {
        progressCircle.setAttribute('data-value', 'red');
    } else {
        progressCircle.removeAttribute('data-value');
    }
}