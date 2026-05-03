document.addEventListener('DOMContentLoaded', () => {
    // Generate background particles
    const particlesContainer = document.getElementById('particles');
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.style.position = 'absolute';
        particle.style.width = Math.random() * 5 + 2 + 'px';
        particle.style.height = particle.style.width;
        particle.style.background = 'rgba(255, 255, 255, 0.3)';
        particle.style.borderRadius = '50%';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animation = `particleFloat ${Math.random() * 10 + 15}s infinite linear`;
        particle.style.animationDelay = Math.random() * 5 + 's';
        particlesContainer.appendChild(particle);
    }

    // Elements
    const radios = document.querySelectorAll('input[name="method"]');
    const uploadForm = document.getElementById('upload-form');
    const webcamForm = document.getElementById('webcam-form');
    const fileInput = document.getElementById('file-input');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    
    const uploadSection = document.getElementById('upload-section');
    const imageDisplaySection = document.getElementById('image-display-section');
    const previewImage = document.getElementById('preview-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsSection = document.getElementById('results-section');
    
    const verifyContent = document.getElementById('verify-content');
    const predictionContent = document.getElementById('prediction-content');
    const reportContent = document.getElementById('report-content');

    let stream = null;
    let selectedFile = null;

    // Toggle Input Method
    radios.forEach(radio => {
        radio.addEventListener('change', async (e) => {
            if (e.target.value === 'upload') {
                uploadForm.style.display = 'block';
                webcamForm.style.display = 'none';
                stopWebcam();
            } else {
                uploadForm.style.display = 'none';
                webcamForm.style.display = 'block';
                startWebcam();
            }
        });
    });

    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (err) {
            console.error("Error accessing webcam: ", err);
            alert("Could not access webcam.");
        }
    }

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    // Handle File Selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            selectedFile = e.target.files[0];
            showPreview(URL.createObjectURL(selectedFile));
        }
    });

    // Handle Webcam Capture
    captureBtn.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
            selectedFile = new File([blob], "capture.jpg", { type: "image/jpeg" });
            showPreview(URL.createObjectURL(selectedFile));
        }, 'image/jpeg');
    });

    function showPreview(url) {
        previewImage.src = url;
        uploadSection.style.display = 'none';
        imageDisplaySection.style.display = 'block';
        resultsSection.style.display = 'none';
    }

    // Reset
    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = "";
        uploadSection.style.display = 'block';
        imageDisplaySection.style.display = 'none';
        resultsSection.style.display = 'none';
        if (document.querySelector('input[name="method"]:checked').value === 'webcam') {
            startWebcam();
        }
    });

    // Analyze
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        imageDisplaySection.style.display = 'none';
        loadingSpinner.style.display = 'block';

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Network response was not ok");
            
            const data = await response.json();
            displayResults(data);
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
            imageDisplaySection.style.display = 'block';
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });

    function extractFloat(str) {
        const match = str.match(/[-+]?\d*\.\d+|\d+/);
        return match ? parseFloat(match[0]) : 10;
    }

    function getSeverityClass(val) {
        return val > 80 ? "severity-high" : val > 50 ? "severity-medium" : "severity-low";
    }

    function displayResults(data) {
        // Parse Verify
        const verifyText = data.verify ? data.verify.replace(/`/g, '') : '';
        const verifyParts = verifyText.split(',');
        
        let verifyHTML = '';
        if (verifyParts.length >= 4) {
            const confVal = extractFloat(verifyParts[1]);
            verifyHTML = `
                <p><strong>🎯 Classification:</strong> ${verifyParts[0]}</p>
                <p><strong>📊 Confidence:</strong> <span class="${getSeverityClass(confVal)}">${verifyParts[1]}</span></p>
                <p><strong>🧬 Skin Type:</strong> ${verifyParts[2]}</p>
                <p><strong>💡 Remarks:</strong> ${verifyParts[3]}</p>
            `;
        } else {
            verifyHTML = `<p>${verifyText}</p>`;
        }
        verifyContent.innerHTML = verifyHTML;

        // Parse Prediction
        const predText = data.prediction || '';
        const predParts = predText.split(',');
        
        let predHTML = '';
        if (predParts.length >= 2) {
            const confVal = extractFloat(predParts[1]);
            predHTML = `
                <p><strong>🏥 Condition:</strong> ${predParts[0]}</p>
                <p><strong>📈 Confidence:</strong> <span class="${getSeverityClass(confVal)}">${predParts[1]}</span></p>
            `;
            if (predParts.length > 2) {
                predHTML += `<p><strong>📝 Analysis:</strong> ${predParts.slice(2).join(',')}</p>`;
            }
        } else {
            predHTML = `<p>${predText}</p>`;
        }
        predictionContent.innerHTML = predHTML;

        // Parse Report
        reportContent.innerHTML = marked.parse(data.report || "No report available.");

        resultsSection.style.display = 'block';
    }
});
