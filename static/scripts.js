// Handle form submission for image upload
document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    let formData = new FormData(this);
    fetch('/upload_image', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
    .then(data => {
        alert(data.message);
        // Display uploaded images and their dimensions
        let uploadedImagesDiv = document.getElementById('uploadedImages');
        uploadedImagesDiv.innerHTML = '';
        data.files.forEach(file => {
            let imgElement = document.createElement('img');
            imgElement.src = `uploads/${file.filename}`;
            imgElement.classList.add('img-fluid', 'rounded', 'shadow-sm', 'mx-auto', 'd-block', 'my-2');
            uploadedImagesDiv.appendChild(imgElement);
            let dimensionText = document.createElement('p');
            dimensionText.textContent = `Filename: ${file.filename}, Width: ${file.width}, Height: ${file.height}`;
            uploadedImagesDiv.appendChild(dimensionText);
        });
    });
});

// Fetch and display histogram for the specified image
function getHistogram() {
    let filename = document.getElementById('filename').value;
    fetch(`/histogram/${filename}`)
        .then(response => response.blob())
        .then(imageBlob => {
            let imageObjectURL = URL.createObjectURL(imageBlob);
            let img = document.getElementById('histogram');
            img.src = imageObjectURL;
            img.style.display = 'block';
        });
}

// Fetch and display segmentation mask for the specified image
function getSegmentation() {
    let filename = document.getElementById('filename').value;
    fetch(`/segmentation/${filename}`)
        .then(response => response.blob())
        .then(imageBlob => {
            let imageObjectURL = URL.createObjectURL(imageBlob);
            let img = document.getElementById('segmentation');
            img.src = imageObjectURL;
            img.style.display = 'block';
        });
}

// Resize the specified image to the given dimensions and display it
function resizeImage() {
    let filename = document.getElementById('filename').value;
    let width = document.getElementById('resizeWidth').value;
    let height = document.getElementById('resizeHeight').value;

    fetch(`/resize/${filename}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({width: parseInt(width), height: parseInt(height)})
    }).then(response => response.blob())
    .then(imageBlob => {
        let imageObjectURL = URL.createObjectURL(imageBlob);
        let img = document.getElementById('resized');
        img.src = imageObjectURL;
        img.style.display = 'block';
    });
}

// Crop the specified image to the given coordinates and display it
function cropImage() {
    let filename = encodeURIComponent(document.getElementById('filename').value);
    let left = parseInt(document.getElementById('cropLeft').value);
    let top = parseInt(document.getElementById('cropTop').value);
    let right = parseInt(document.getElementById('cropRight').value);
    let bottom = parseInt(document.getElementById('cropBottom').value);

    if (isNaN(left) || isNaN(top) || isNaN(right) || isNaN(bottom)) {
        console.error('Invalid crop coordinates');
        return;
    }

    fetch(`/crop/${filename}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            left: left,
            top: top,
            right: right,
            bottom: bottom
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.blob();
    })
    .then(blob => {
        let url = URL.createObjectURL(blob);
        document.getElementById('cropped').src = url;
        document.getElementById('cropped').style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
}

// Convert the specified image to the selected format and display it
function convertImage() {
    let filename = document.getElementById('filename').value;
    let format = document.getElementById('convertFormat').value.toLowerCase();

    fetch(`/convert/${filename}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({format: format})
    }).then(response => response.blob())
    .then(imageBlob => {
        let imageObjectURL = URL.createObjectURL(imageBlob);
        let img = document.getElementById('converted');
        img.src = imageObjectURL;
        img.style.display = 'block';
    });
}
