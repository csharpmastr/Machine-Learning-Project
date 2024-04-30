        // Function to get URL parameter by name
        function getUrlParameter(name) {
            name = name.replace(/[\[]/, '\\[').replace(/[\]]/, '\\]');
            var regex = new RegExp('[\\?&]' + name + '=([^&#]*)');
            var results = regex.exec(location.search);
            return results === null ? '' : decodeURIComponent(results[1].replace(/\+/g, ' '));
        };

        // Get the image URL parameter from the URL
        var imageUrl = getUrlParameter('imageUrl');

        // Display the image
        if (imageUrl) {
            var imageContainer = document.getElementById('imageContainer');
            var img = document.createElement('img');
            img.src = imageUrl;
            img.alt = 'Uploaded Image';
            imageContainer.appendChild(img);
        } else {
            // Handle case where no image URL parameter is provided
            console.error('No image URL parameter found.');
        }
        