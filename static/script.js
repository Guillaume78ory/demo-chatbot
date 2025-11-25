document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const themeToggleBtn = document.getElementById('theme-toggle');
    let chatHistory = [];

    const appWrapper = document.querySelector('.app-wrapper');
    const historyToggleBtn = document.getElementById('history-toggle');
    const historyList = document.getElementById('history-list');
    const closeHistoryBtn = document.getElementById('close-history');
    
    const fileInput = document.getElementById('file-input');
    const attachBtn = document.getElementById('attach-btn');

    // --- GESTION THEME ---
    function enableDarkMode() {
        document.body.classList.remove('light-mode');
        document.body.classList.add('dark-mode');
        localStorage.setItem('theme', 'dark');
    }

    function enableLightMode() {
        document.body.classList.remove('dark-mode');
        document.body.classList.add('light-mode');
        localStorage.setItem('theme', 'light');
    }

    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') enableLightMode();
    else enableDarkMode();

    themeToggleBtn.addEventListener('click', () => {
        if (document.body.classList.contains('dark-mode')) enableLightMode();
        else enableDarkMode();
    });

    // --- HISTORIQUE ---
    historyToggleBtn.addEventListener('click', () => appWrapper.classList.toggle('history-open'));
    if (closeHistoryBtn) closeHistoryBtn.addEventListener('click', () => appWrapper.classList.remove('history-open'));

    function addHistoryItem(text, messageId) {
        const li = document.createElement('li');
        li.textContent = text.length > 40 ? text.substring(0, 40) + '...' : text;
        li.title = text;
        li.addEventListener('click', () => {
            const target = document.getElementById(messageId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                target.classList.add('highlight');
                setTimeout(() => target.classList.remove('highlight'), 2000);
            }
            if (window.innerWidth <= 768) appWrapper.classList.remove('history-open');
        });
        historyList.prepend(li);
    }

    // --- UPLOAD ---
    if (attachBtn && fileInput) {
        attachBtn.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', async () => {
            const file = fileInput.files[0];
            if (!file) return;
            
            addMessage(`📤 Analyse de "${file.name}"...`, 'bot');
            const formData = new FormData();
            formData.append('file', file);

            try {
                const resp = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await resp.json();
                addMessage(data.status === 'success' ? `✅ ${data.message}` : `❌ Erreur: ${data.message}`, 'bot');
            } catch (e) {
                addMessage("❌ Erreur réseau upload.", 'bot');
            }
            fileInput.value = ''; 
        });
    }

    // --- CHAT ---
    function escapeHtml(str) {
        if (!str) return '';
        return String(str).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    function addMessage(message, sender, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        const messageId = 'msg-' + Date.now();
        messageDiv.id = messageId;
        
        let inner = `<p>${escapeHtml(message).replace(/\n/g, '<br>')}</p>`;
        
        if (sources && sources.length && sender !== 'user') {
            inner += '<br><strong>Sources :</strong><ul>';
            const grouped = {};
            sources.forEach(s => {
                const doc = s['Document'] || 'Inconnu';
                const page = s['Page/Feuille'] || 'N/A';
                if (!grouped[doc]) grouped[doc] = new Set();
                grouped[doc].add(page);
            });
            
            for (const [doc, pages] of Object.entries(grouped)) {
                inner += `<li>${escapeHtml(doc)} (p. ${Array.from(pages).join(', ')})</li>`;
            }
            inner += '</ul>';
        }

        messageDiv.innerHTML = inner;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        if (sender === 'user') addHistoryItem(message, messageId);
    }
    
    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        addMessage(query, 'user');
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;

        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot-message';
        loadingDiv.innerHTML = `<div class="loading-message-container"><div class="inline-spinner"></div><p><i>L'assistant réfléchit<span class="loading-dot">.</span><span class="loading-dot">.</span><span class="loading-dot">.</span></i></p></div>`;
        chatBox.appendChild(loadingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, chat_history: chatHistory })
            });
            const data = await resp.json();
            loadingDiv.remove();
            addMessage(data.answer || '', 'bot', data.sources);
            chatHistory.push({ role: 'user', content: query });
            chatHistory.push({ role: 'assistant', content: data.answer || '' });
        } catch (e) {
            loadingDiv.remove();
            addMessage("Erreur technique.", 'bot');
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    if (chatForm) {
        chatForm.addEventListener('submit', (e) => { e.preventDefault(); sendMessage(); });
    }
});