const API_BASE = 'http://127.0.0.1:8800/api/v1';

// DOM元素 - 将在 DOMContentLoaded 后初始化
let chatMessages;
let questionInput;
let sendBtn;
let uploadBtn;
let uploadModal;
let closeModal;
let fileInput;
let confirmUploadBtn;
let webSearchToggle;
let dirSelect;
let sidebar;
let toggleSidebar;
let newChatBtn;
let chatList;

// 状态管理
let isStreaming = false;
let selectedFiles = [];
let chats = []; // 所有对话
let currentChatId = null; // 当前对话ID
let chatCounter = 0;

// 初始化DOM元素
function initDOMElements() {
    chatMessages = document.getElementById('chatMessages');
    questionInput = document.getElementById('questionInput');
    sendBtn = document.getElementById('sendBtn');
    uploadBtn = document.getElementById('uploadBtn');
    uploadModal = document.getElementById('uploadModal');
    closeModal = document.querySelector('.close');
    fileInput = document.getElementById('modalFileInput');
    confirmUploadBtn = document.getElementById('confirmUploadBtn');
    webSearchToggle = document.getElementById('webSearchToggle');
    dirSelect = document.getElementById('dirSelect');
    sidebar = document.getElementById('sidebar');
    toggleSidebar = document.getElementById('toggleSidebar');
    newChatBtn = document.getElementById('newChatBtn');
    chatList = document.getElementById('chatList');
    
    // 检查关键元素
    if (!confirmUploadBtn) {
        console.error('confirmUploadBtn not found!');
    }
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initDOMElements();
    setupEventListeners();
    loadKnowledgeBases();
    createNewChat(); // 创建第一个对话
});

function setupEventListeners() {
    // 发送按钮
    sendBtn.addEventListener('click', sendMessage);
    
    // 回车发送
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 输入框变化
    questionInput.addEventListener('input', () => {
        sendBtn.disabled = !questionInput.value.trim() || isStreaming;
        // 自动调整高度
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    });

    // 上传按钮
    uploadBtn.addEventListener('click', () => {
        uploadModal.style.display = 'block';
        // 重置状态
        selectedFiles = [];
        fileInput.value = '';
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('uploadProgress').innerHTML = '';
        document.getElementById('confirmUploadBtn').disabled = true;
    });

    // 侧边栏切换
    toggleSidebar.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });

    // 新建对话
    newChatBtn.addEventListener('click', createNewChat);

    // 模态框关闭
    closeModal.addEventListener('click', () => {
        uploadModal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === uploadModal) {
            uploadModal.style.display = 'none';
        }
    });

    // 文件选择
    fileInput.addEventListener('change', (e) => {
        selectedFiles = Array.from(e.target.files);
        showFileInfo();
        // 启用上传按钮
        document.getElementById('confirmUploadBtn').disabled = selectedFiles.length === 0;
    });

    // 确认上传
    if (confirmUploadBtn) {
        confirmUploadBtn.addEventListener('click', uploadFiles);
        console.log('confirmUploadBtn event listener added');
    } else {
        console.error('Cannot add event listener: confirmUploadBtn is null');
    }
}

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question || isStreaming || !currentChatId) return;

    // 获取当前对话
    const currentChat = chats.find(c => c.id === currentChatId);
    if (!currentChat) return;

    // 添加到界面和聊天记录
    displayMessage('user', question);
    addMessageToChat('user', question);
    questionInput.value = '';
    sendBtn.disabled = true;
    questionInput.style.height = 'auto';
    isStreaming = true;

    // 显示加载指示器
    const messageDiv = addMessageForStreaming('assistant', '');
    const indicator = showTypingIndicator(messageDiv);
    
    try {
        // 构建历史对话
        const historyPairs = [];
        const messages = currentChat.messages.filter(m => m.role !== 'system');
        
        for (let i = 0; i < messages.length; i += 2) {
            if (messages[i] && messages[i + 1] && messages[i].role === 'user') {
                historyPairs.push([messages[i].content, messages[i + 1].content || '']);
            }
        }

        // 调用API
        const requestData = {
            question: question,
            history: historyPairs,
            is_web_search: webSearchToggle.checked,
            dir_path: dirSelect.value || ""
        };
        
        console.log('Sending request:', requestData);
        
        const response = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`请求失败: ${response.status}`);
        }

        const data = await response.json();
        indicator.remove();
        
        if (data.answer) {
            displayMessage('assistant', data.answer);
            addMessageToChat('assistant', data.answer);
        } else {
            displayMessage('assistant', '抱歉，我没有获取到答案。');
            addMessageToChat('assistant', '抱歉，我没有获取到答案。');
        }

        // 更新对话标题
        updateChatTitle(currentChatId, question);

    } catch (error) {
        indicator.remove();
        const errorMsg = `❌ 错误: ${error.message}`;
        displayMessage('assistant', errorMsg);
        addMessageToChat('assistant', errorMsg);
        console.error('Error:', error);
    } finally {
        isStreaming = false;
        sendBtn.disabled = false;
    }
}

function createNewChat() {
    const chatId = `chat_${Date.now()}`;
    const chat = {
        id: chatId,
        title: `新对话 ${++chatCounter}`,
        messages: [],
        createdAt: Date.now()
    };
    
    chats.push(chat);
    currentChatId = chatId;
    
    renderChatList();
    loadChat(chatId);
}

function loadChat(chatId) {
    currentChatId = chatId;
    const chat = chats.find(c => c.id === chatId);
    if (!chat) return;

    // 清空消息区域
    chatMessages.innerHTML = '';
    
    // 如果没有消息，显示欢迎消息
    if (chat.messages.length === 0) {
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <p>👋 开始新的对话</p>
            </div>
        `;
    } else {
        // 重新渲染所有消息
        chat.messages.forEach(msg => {
            displayMessage(msg.role, msg.content);
        });
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    renderChatList();
}

function addMessageToChat(role, content) {
    if (!currentChatId) return;
    
    const currentChat = chats.find(c => c.id === currentChatId);
    if (currentChat) {
        currentChat.messages.push({ role, content });
    }
}

function displayMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    if (role === 'assistant') {
        bubble.innerHTML = formatMarkdown(content);
    } else {
        bubble.textContent = content;
    }
    
    messageDiv.appendChild(bubble);
    chatMessages.appendChild(messageDiv);
    
    // 移除欢迎消息
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
}

function addMessage(role, content) {
    // 添加到界面
    displayMessage(role, content);
    
    // 添加到当前对话记录
    addMessageToChat(role, content);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderChatList() {
    chatList.innerHTML = '';
    
    chats.sort((a, b) => b.createdAt - a.createdAt).forEach(chat => {
        const item = document.createElement('div');
        item.className = `chat-item ${chat.id === currentChatId ? 'active' : ''}`;
        item.onclick = () => loadChat(chat.id);
        
        item.innerHTML = `
            <div class="chat-item-title">${chat.title}</div>
            <div class="chat-item-actions">
                <button class="delete-btn" onclick="event.stopPropagation(); deleteChat('${chat.id}')">🗑️</button>
            </div>
        `;
        
        chatList.appendChild(item);
    });
}

function deleteChat(chatId) {
    chats = chats.filter(c => c.id !== chatId);
    
    if (currentChatId === chatId) {
        // 切换到其他对话或创建新对话
        if (chats.length > 0) {
            currentChatId = chats[0].id;
            loadChat(currentChatId);
        } else {
            createNewChat();
        }
    }
    
    renderChatList();
}

function updateChatTitle(chatId, question) {
    const chat = chats.find(c => c.id === chatId);
    if (chat && chat.messages.length === 2) {
        // 只有第一条消息时更新标题
        chat.title = question.length > 20 ? question.substring(0, 20) + '...' : question;
        renderChatList();
    }
}

function addMessageForStreaming(role, content) {
    if (content === '' && role === 'assistant') {
        // 返回占位符元素用于后续更新
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        chatMessages.appendChild(messageDiv);
        return messageDiv;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    if (role === 'assistant') {
        bubble.innerHTML = formatMarkdown(content);
    } else {
        bubble.textContent = content;
    }
    
    messageDiv.appendChild(bubble);
    chatMessages.appendChild(messageDiv);
    
    // 移除欢迎消息
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function showTypingIndicator(parentDiv) {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    parentDiv.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return indicator;
}

function formatMarkdown(text) {
    // 简单的Markdown格式化
    return text
        .replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

function clearChat() {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <p>👋 欢迎使用智能文档问答系统！</p>
            <p>您可以：</p>
            <ul>
                <li>上传文档到知识库</li>
                <li>提问获取答案</li>
                <li>开启网络搜索获取最新信息</li>
            </ul>
        </div>
    `;
    currentChat = [];
}

function showFileInfo() {
    const fileInfo = document.getElementById('fileInfo');
    const confirmBtn = confirmUploadBtn || document.getElementById('confirmUploadBtn');
    
    if (!confirmBtn) {
        console.error('confirmUploadBtn not found in showFileInfo');
        return;
    }
    
    if (selectedFiles.length === 0) {
        fileInfo.style.display = 'none';
        confirmBtn.disabled = true;
        return;
    }
    
    fileInfo.style.display = 'block';
    fileInfo.innerHTML = `
        <p><strong>已选择 ${selectedFiles.length} 个文件:</strong></p>
        <ul>
            ${selectedFiles.map(f => `<li>📄 ${f.name} (${(f.size / 1024).toFixed(2)} KB)</li>`).join('')}
        </ul>
    `;
    
    // 启用上传按钮
    confirmBtn.disabled = false;
    confirmBtn.style.opacity = '1';
    confirmBtn.style.cursor = 'pointer';
    confirmBtn.style.visibility = 'visible';
    confirmBtn.style.display = 'inline-block';
    
    console.log('Files selected:', selectedFiles.length, 'Button disabled:', confirmBtn.disabled);
    console.log('Button element:', confirmBtn);
    console.log('Button style display:', getComputedStyle(confirmBtn).display);
    console.log('Button style visibility:', getComputedStyle(confirmBtn).visibility);
}

async function uploadFiles(event) {
    event?.preventDefault();
    
    console.log('uploadFiles called, selectedFiles:', selectedFiles.length);
    
    if (selectedFiles.length === 0) {
        alert('请选择要上传的文件');
        return;
    }

    const progressDiv = document.getElementById('uploadProgress');
    progressDiv.innerHTML = '<p style="text-align: center;">⏳ 正在上传...</p>';
    progressDiv.style.display = 'block';

    try {
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        console.log('Uploading files to:', `${API_BASE}/upload`);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        console.log('Upload response status:', response.status);

        const result = await response.json();

        if (response.ok) {
            let successMsg = `<p style="color: green; text-align: center;">✅ 上传成功 ${result.uploaded.length} 个文件</p>`;
            if (result.failed && result.failed.length > 0) {
                successMsg += `<p style="color: orange; text-align: center;">⚠️ ${result.failed.length} 个文件上传失败</p>`;
            }
            progressDiv.innerHTML = successMsg;
            
            // 重新加载知识库列表
            await loadKnowledgeBases();
            
            // 清空文件选择
            selectedFiles = [];
            fileInput.value = '';
            document.getElementById('fileInfo').style.display = 'none';
            document.getElementById('confirmUploadBtn').disabled = true;
            
            setTimeout(() => {
                uploadModal.style.display = 'none';
                progressDiv.innerHTML = '';
            }, 2000);
        } else {
            progressDiv.innerHTML = `<p style="color: red; text-align: center;">❌ 上传失败: ${result.detail || '未知错误'}</p>`;
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        progressDiv.innerHTML = `<p style="color: red; text-align: center;">❌ 上传失败: ${error.message}</p>`;
    }
}

async function loadKnowledgeBases() {
    try {
        const response = await fetch(`${API_BASE}/knowledge-bases`);
        const data = await response.json();
        
        let options = '<option value="">默认知识库</option>';
        
        if (data.knowledge_bases && data.knowledge_bases.length > 0) {
            options += data.knowledge_bases
                .map(kb => `<option value="${kb.name}">${kb.name} (${kb.file_count} 文件)</option>`)
                .join('');
        }
        
        dirSelect.innerHTML = options;
    } catch (error) {
        console.error('加载知识库失败:', error);
        // 默认选项
        dirSelect.innerHTML = '<option value="">默认知识库</option>';
    }
}

