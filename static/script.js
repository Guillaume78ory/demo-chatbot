document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    
    // Boutons
    const themeToggleBtn = document.getElementById('theme-toggle');
    const historyToggleBtn = document.getElementById('history-toggle');
    // Le bouton de fermeture de la sidebar
    const closeHistoryBtn = document.getElementById('close-history'); 
    
    const appWrapper = document.querySelector('.app-wrapper');
    const historyList = document.getElementById('history-list');
    
    let chatHistory = [];

    // --- GESTION DU THÈME ---
    
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

    // --- INITIALISATION DU THÈME ---
    const savedTheme = localStorage.getItem('theme');

    // Si l'utilisateur a explicitement choisi le mode CLAIR, on le met
    if (savedTheme === 'light') {
        enableLightMode();
    } 
    // Dans TOUS les autres cas (première visite ou 'dark'), on met le sombre
    else {
        enableDarkMode();
    }

    // Bascule du thème au clic
    themeToggleBtn.addEventListener('click', () => {
        if (document.body.classList.contains('dark-mode')) {
            enableLightMode();
        } else {
            enableDarkMode();
        }
    });

    // --- GESTION DE L'HISTORIQUE ---

    // Ouvrir/Fermer avec le bouton "Horloge"
    historyToggleBtn.addEventListener('click', () => {
        appWrapper.classList.toggle('history-open');
    });

    // Fermer avec le bouton "X" dans la sidebar
    if (closeHistoryBtn) {
        closeHistoryBtn.addEventListener('click', () => {
            appWrapper.classList.remove('history-open');
        });
    }

    // --- FONCTIONS ADDITIONNELLES ---

    function addHistoryItem(text, messageId) {
        const li = document.createElement('li');
        li.textContent = text.length > 40 ? text.substring(0, 40) + '...' : text;
        li.title = text; 
        
        li.addEventListener('click', () => {
            const targetMessage = document.getElementById(messageId);
            if (targetMessage) {
                targetMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
                targetMessage.classList.add('highlight');
                setTimeout(() => {
                    targetMessage.classList.remove('highlight');
                }, 2000);
            }
            if (window.innerWidth <= 768) {
                appWrapper.classList.remove('history-open');
            }
        });

        historyList.prepend(li);
    }

    function isRefusal(message) {
        const lowerCaseMessage = message.toLowerCase();
        const refusalPhrases = [
            "i cannot answer", "i can't answer", "i do not know", "i don't know",
            "based on the provided context", "the provided documents do not mention",
            "je ne peux pas répondre", "je ne sais pas",
            "d'après le contexte fourni", "pas d'information", "je suis désolé"
        ];
        return refusalPhrases.some(phrase => lowerCaseMessage.includes(phrase));
    }

    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str).replace(/[&<>"]/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'
        }[c]));
    }

    function addMessage(message, sender, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        
        const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        messageDiv.id = messageId;
        
        const formattedMessage = escapeHtml(message).replace(/\n/g, '<br>');
        let inner = `<p>${formattedMessage}</p>`;
        
        if (sources && Array.isArray(sources) && sources.length && !isRefusal(message)) {
            const grouped = groupSources(sources);
            inner += '<br><strong>Sources utilisées :</strong><ul>';
            grouped.forEach(g => {
                inner += `<li>${escapeHtml(g.doc)} — p.${escapeHtml(g.pages.join(', '))}</li>`;
            });
            inner += '</ul>';
        }

        messageDiv.innerHTML = inner;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        if (sender === 'user') {
            addHistoryItem(message, messageId);
        }
    }
    
    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        addMessage(query, 'user');
        userInput.value = '';
        
        userInput.disabled = true;
        sendBtn.disabled = true;

        const loadingMessageElement = document.createElement('div');
        loadingMessageElement.classList.add('message', 'bot-message');
        loadingMessageElement.innerHTML = `
            <div class="loading-message-container">
                <div class="inline-spinner"></div>
                <p><i>L'assistant réfléchit...</i></p>
            </div>
        `;
        chatBox.appendChild(loadingMessageElement);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, chat_history: chatHistory })
            });
            if (!resp.ok) throw new Error(`Erreur réseau : ${resp.statusText}`);
            
            const data = await resp.json();
            
            loadingMessageElement.remove(); 
            addMessage(data.answer || '', 'bot', data.sources);
            chatHistory.push({ role: 'user', content: query });
            chatHistory.push({ role: 'assistant', content: data.answer || '' });
            
        } catch (error) {
            console.error('Erreur lors de la requête :', error);
            loadingMessageElement.remove(); 
            addMessage("Désolé, une erreur est survenue. Veuillez réessayer.", 'bot');
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    if (chatForm) {
        chatForm.addEventListener('submit', (event) => {
            event.preventDefault(); 
            sendMessage();
        });
    } else {
        console.error("Formulaire introuvable.");
    }

    // --- FONCTIONS UTILITAIRES ---
    function safeGetPage(meta) {if (!meta) return null; return meta['Page/Feuille'] || meta['page/feuille'] || meta.page || meta.page_number || meta.pageno || null;}
    function safeGetDoc(meta) {if (!meta) return 'inconnu'; return meta.Document || meta.document || meta.source || meta.file || meta.filename || meta.path || 'inconnu';}
    function groupSources(sources) {const map = new Map(); (sources || []).forEach(s => {try {const doc = safeGetDoc(s) || 'inconnu'; const page = safeGetPage(s) || '?'; if (!map.has(doc)) map.set(doc, new Set()); map.get(doc).add(String(page));} catch (e) {console.warn('Erreur dans groupSources', s, e);}}); const out = []; for (const [doc, pagesSet] of map.entries()) {const pages = Array.from(pagesSet).sort((a,b) => {if (a === '?') return 1; if (b === '?') return -1; const na = Number(a), nb = Number(b); if (isNaN(na) || isNaN(nb)) return a.localeCompare(b); return na - nb;}); out.push({ doc, pages });} return out;}
});