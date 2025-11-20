document.addEventListener('DOMContentLoaded', () => {
    // --- SÉLECTEURS DOM ---
    const chatForm = document.getElementById('chat-form');
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    
    // Boutons de l'interface
    const themeToggleBtn = document.getElementById('theme-toggle');
    const historyToggleBtn = document.getElementById('history-toggle');
    const closeHistoryBtn = document.getElementById('close-history'); // Le nouveau bouton
    
    // Éléments de structure
    const appWrapper = document.querySelector('.app-wrapper');
    const historyList = document.getElementById('history-list');
    
    let chatHistory = [];

    // --- GESTION DU THÈME (DARK MODE PAR DÉFAUT) ---
    
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

    // Logique d'initialisation du thème
    const savedTheme = localStorage.getItem('theme');

    if (savedTheme === 'light') {
        // Si l'utilisateur a explicitement choisi le mode clair auparavant
        enableLightMode();
    } else {
        // Sinon (si c'est 'dark' OU si c'est la première visite null), on met le mode sombre
        enableDarkMode();
    }

    // Écouteur pour le bouton de changement de thème
    themeToggleBtn.addEventListener('click', () => {
        if (document.body.classList.contains('dark-mode')) {
            enableLightMode();
        } else {
            enableDarkMode();
        }
    });

    // --- GESTION DE L'HISTORIQUE (SIDEBAR) ---

    // Ouvrir/Fermer avec le bouton du header
    historyToggleBtn.addEventListener('click', () => {
        appWrapper.classList.toggle('history-open');
    });

    // Fermer avec le nouveau bouton "X" dans la sidebar
    if (closeHistoryBtn) {
        closeHistoryBtn.addEventListener('click', () => {
            appWrapper.classList.remove('history-open');
        });
    }

    // Fonction pour ajouter un élément à la liste d'historique
    function addHistoryItem(text, messageId) {
        const li = document.createElement('li');
        // Tronque le texte s'il est trop long (40 caractères)
        li.textContent = text.length > 40 ? text.substring(0, 40) + '...' : text;
        li.title = text; // Affiche tout le texte au survol
        
        li.addEventListener('click', () => {
            const targetMessage = document.getElementById(messageId);
            if (targetMessage) {
                // Scroll vers le message
                targetMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
                
                // Effet de surlignage temporaire
                targetMessage.classList.add('highlight');
                setTimeout(() => {
                    targetMessage.classList.remove('highlight');
                }, 2000);
                
                // Sur mobile, on ferme le menu après le clic pour voir le message
                if (window.innerWidth <= 768) {
                    appWrapper.classList.remove('history-open');
                }
            }
        });

        // Ajoute le nouvel élément en haut de la liste
        historyList.prepend(li);
    }

    // --- GESTION DU CHAT ---

    // Détection des réponses de refus (pour ne pas afficher les sources)
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

    // Fonction utilitaire pour échapper le HTML (sécurité)
    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str).replace(/[&<>"]/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'
        }[c]));
    }

    // Fonction pour afficher un message dans la chat box
    function addMessage(message, sender, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        
        // ID unique pour le lien avec l'historique
        const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        messageDiv.id = messageId;
        
        const formattedMessage = escapeHtml(message).replace(/\n/g, '<br>');
        let inner = `<p>${formattedMessage}</p>`;
        
        // Ajout des sources si disponibles
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
        
        // Auto-scroll vers le bas
        chatBox.scrollTop = chatBox.scrollHeight;

        // Si c'est un message utilisateur, on l'ajoute à l'historique
        if (sender === 'user') {
            addHistoryItem(message, messageId);
        }
    }
    
    // Fonction d'envoi du message vers le backend
    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        // 1. Afficher le message utilisateur
        addMessage(query, 'user');
        userInput.value = '';
        
        // 2. Désactiver l'input pendant le chargement
        userInput.disabled = true;
        sendBtn.disabled = true;

        // 3. Afficher l'indicateur de chargement
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
            // 4. Appel API
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, chat_history: chatHistory })
            });
            
            if (!resp.ok) throw new Error(`Erreur réseau : ${resp.statusText}`);
            
            const data = await resp.json();
            
            // 5. Supprimer le chargement et afficher la réponse
            loadingMessageElement.remove(); 
            addMessage(data.answer || '', 'bot', data.sources);
            
            // 6. Mettre à jour l'historique local pour le contexte
            chatHistory.push({ role: 'user', content: query });
            chatHistory.push({ role: 'assistant', content: data.answer || '' });
            
        } catch (error) {
            console.error('Erreur lors de la requête :', error);
            loadingMessageElement.remove(); 
            addMessage("Désolé, une erreur technique est survenue. Veuillez réessayer.", 'bot');
        } finally {
            // 7. Réactiver l'input
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    // Gestionnaire de soumission du formulaire
    if (chatForm) {
        chatForm.addEventListener('submit', (event) => {
            event.preventDefault(); 
            sendMessage();
        });
    } else {
        console.error("ERREUR CRITIQUE : Le formulaire avec l'ID 'chat-form' est introuvable.");
    }

    // --- FONCTIONS UTILITAIRES ---
    function safeGetPage(meta) {
        if (!meta) return null; 
        return meta['Page/Feuille'] || meta['page/feuille'] || meta.page || meta.page_number || meta.pageno || null;
    }
    
    function safeGetDoc(meta) {
        if (!meta) return 'inconnu'; 
        return meta.Document || meta.document || meta.source || meta.file || meta.filename || meta.path || 'inconnu';
    }
    
    function groupSources(sources) {
        const map = new Map(); 
        (sources || []).forEach(s => {
            try {
                const doc = safeGetDoc(s) || 'inconnu'; 
                const page = safeGetPage(s) || '?'; 
                if (!map.has(doc)) map.set(doc, new Set()); 
                map.get(doc).add(String(page));
            } catch (e) {
                console.warn('Erreur dans groupSources', s, e);
            }
        }); 
        const out = []; 
        for (const [doc, pagesSet] of map.entries()) {
            const pages = Array.from(pagesSet).sort((a,b) => {
                if (a === '?') return 1; 
                if (b === '?') return -1; 
                const na = Number(a), nb = Number(b); 
                if (isNaN(na) || isNaN(nb)) return a.localeCompare(b); 
                return na - nb;
            }); 
            out.push({ doc, pages });
        } 
        return out;
    }
});