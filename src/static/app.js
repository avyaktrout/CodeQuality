/**
 * Bug Prediction - Frontend Application
 */

const API_BASE = '';

// Store examples globally for click handlers
let examplesData = [];

// Load examples on page load
document.addEventListener('DOMContentLoaded', () => {
    loadExamples();
});

/**
 * Analyze the code in the textarea
 */
async function analyzeCode() {
    const codeInput = document.getElementById('code-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const resultContent = document.getElementById('result-content');

    const code = codeInput.value.trim();

    if (!code) {
        showError('Please enter some Python code to analyze.');
        return;
    }

    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    resultSection.classList.remove('hidden');
    resultContent.innerHTML = '<div class="loading" style="height: 100px;"></div>';

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ code }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }

        const result = await response.json();
        displayResult(result);
    } catch (error) {
        showError(error.message);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Code';
    }
}

/**
 * Display the prediction result
 */
function displayResult(result) {
    const resultContent = document.getElementById('result-content');
    const predictionClass = result.prediction.toLowerCase();
    const probabilityPercent = (result.probability * 100).toFixed(1);
    const confidencePercent = (result.confidence * 100).toFixed(0);

    let featuresHtml = '';
    if (result.top_features && result.top_features.length > 0) {
        const featureTags = result.top_features
            .map(f => `<span class="feature-tag">${f.name}: ${f.value}</span>`)
            .join('');
        featuresHtml = `
            <div class="features-list">
                <h3>Top Contributing Features:</h3>
                ${featureTags}
            </div>
        `;
    }

    resultContent.innerHTML = `
        <div class="result-card">
            <div>
                <span class="prediction-badge ${predictionClass}">
                    ${result.prediction}
                </span>
            </div>

            <div class="probability-bar">
                <div class="probability-marker" style="left: ${probabilityPercent}%"></div>
            </div>

            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">${probabilityPercent}%</div>
                    <div class="metric-label">Bug Probability</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${confidencePercent}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
            </div>

            ${featuresHtml}
        </div>
    `;
}

/**
 * Show an error message
 */
function showError(message) {
    const resultSection = document.getElementById('result-section');
    const resultContent = document.getElementById('result-content');

    resultSection.classList.remove('hidden');
    resultContent.innerHTML = `
        <div class="error">
            <strong>Error:</strong> ${escapeHtml(message)}
        </div>
    `;
}

/**
 * Clear the code input and results
 */
function clearCode() {
    document.getElementById('code-input').value = '';
    document.getElementById('result-section').classList.add('hidden');
}

/**
 * Load example code snippets
 */
async function loadExamples() {
    const container = document.getElementById('examples-container');

    try {
        const response = await fetch(`${API_BASE}/examples`);
        if (!response.ok) {
            throw new Error('Failed to load examples');
        }

        const examples = await response.json();
        displayExamples(examples);
    } catch (error) {
        container.innerHTML = '<p class="error">Failed to load examples</p>';
    }
}

/**
 * Display example cards
 */
function displayExamples(examples) {
    const container = document.getElementById('examples-container');
    examplesData = examples;

    container.innerHTML = examples.map((example, index) => `
        <div class="example-card" data-index="${index}">
            <h4>
                ${escapeHtml(example.name)}
                <span class="expected ${example.expected.toLowerCase()}">${example.expected}</span>
            </h4>
            <p>${escapeHtml(example.description)}</p>
        </div>
    `).join('');

    // Add click handlers using event delegation
    container.querySelectorAll('.example-card').forEach(card => {
        card.addEventListener('click', () => {
            const index = parseInt(card.dataset.index);
            loadExample(examplesData[index].code);
        });
    });
}

/**
 * Load an example into the code input
 */
function loadExample(code) {
    document.getElementById('code-input').value = code;
    document.getElementById('result-section').classList.add('hidden');

    // Scroll to top of input
    document.getElementById('code-input').scrollIntoView({
        behavior: 'smooth',
        block: 'center'
    });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Escape JavaScript string for use in onclick handlers
 */
function escapeJs(text) {
    return text
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'")
        .replace(/"/g, '\\"')
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r')
        .replace(/\t/g, '\\t');
}

// Allow Tab key in textarea
document.getElementById('code-input').addEventListener('keydown', function(e) {
    if (e.key === 'Tab') {
        e.preventDefault();
        const start = this.selectionStart;
        const end = this.selectionEnd;
        this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
        this.selectionStart = this.selectionEnd = start + 4;
    }
});

// Keyboard shortcut: Ctrl+Enter to analyze
document.getElementById('code-input').addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        analyzeCode();
    }
});
