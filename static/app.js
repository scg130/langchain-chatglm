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
let selectedFiles = [];
let chats = []; // 所有对话
let currentChatId = null; // 当前对话ID
let chatCounter = 0;

// 获取当前对话的发送状态
function isCurrentChatStreaming() {
    if (!currentChatId) return false;
    const chat = chats.find(c => c.id === currentChatId);
    return chat ? (chat.isStreaming || false) : false;
}

// 设置当前对话的发送状态
function setCurrentChatStreaming(value) {
    if (!currentChatId) return;
    const chat = chats.find(c => c.id === currentChatId);
    if (chat) {
        chat.isStreaming = value;
    }
}

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
    
    // 页面卸载前保存所有对话的草稿
    window.addEventListener('beforeunload', () => {
        saveCurrentChatDraft();
    });
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

    // 输入框变化 - 自动保存到当前对话的草稿
    questionInput.addEventListener('input', () => {
        // 实时保存输入框内容到当前对话
        saveCurrentChatDraft();
        
        sendBtn.disabled = !questionInput.value.trim() || isCurrentChatStreaming();
        // 自动调整高度
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    });
    
    // 添加全局错误处理，防止请求失败后状态卡住
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
        if (isCurrentChatStreaming()) {
            console.log('Recovering from unhandled rejection...');
            setCurrentChatStreaming(false);
            updateUIForCurrentChat();
        }
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
    if (!question || isCurrentChatStreaming() || !currentChatId) return;

    // 保存发送时的对话ID（防止切换对话后回答显示到错误位置）
    const sendingChatId = currentChatId;
    
    // 获取当前对话
    const currentChat = chats.find(c => c.id === sendingChatId);
    if (!currentChat) return;

    // 添加到界面和聊天记录（使用发送时的对话ID）
    displayMessageToChat('user', question, sendingChatId);
    
    // 清空输入框并保存（清空草稿）
    questionInput.value = '';
    if (currentChat) {
        currentChat.draft = '';
    }
    
    // 设置发送时的对话为发送状态（只阻塞该对话）
    const sendingChat = chats.find(c => c.id === sendingChatId);
    if (sendingChat) {
        sendingChat.isStreaming = true;
    }
    updateUIForCurrentChat();

    // 显示加载指示器（在发送时的对话中）
    const messageDiv = addMessageForStreamingToChat('assistant', '', sendingChatId);
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
        
        // 添加超时控制（60秒）
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
            console.log('Request timeout after 60s');
        }, 60000);
        
        let response;
        try {
            response = await fetch(`${API_BASE}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData),
                signal: controller.signal
            });
        } catch (fetchError) {
            clearTimeout(timeoutId);
            if (fetchError.name === 'AbortError') {
                throw new Error('请求超时（超过60秒），请稍后重试或简化问题');
            }
            throw fetchError;
        }
        
        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `请求失败: ${response.status}`);
        }

        const data = await response.json();
        
        // 移除指示器（在发送时的对话中）
        try {
            if (indicator && indicator.parentNode) {
                indicator.remove();
            }
        } catch (e) {
            console.warn('Could not remove indicator:', e);
        }
        
        // 将回答添加到发送时的对话（不是当前显示的对话）
        if (data.answer) {
            displayMessageToChat('assistant', data.answer, sendingChatId);
        } else {
            displayMessageToChat('assistant', '抱歉，我没有获取到答案。', sendingChatId);
        }

        // 更新对话标题（使用发送时的对话ID）
        updateChatTitle(sendingChatId, question);

    } catch (error) {
        // 安全地移除指示器
        try {
            if (indicator && indicator.parentNode) {
                indicator.remove();
            }
        } catch (e) {
            console.warn('Could not remove indicator:', e);
        }
        
        // 显示错误消息（添加到发送时的对话）
        let errorMsg = '请求失败，请稍后重试';
        if (error.message) {
            errorMsg = `❌ ${error.message}`;
        }
        
        displayMessageToChat('assistant', errorMsg, sendingChatId);
        console.error('Error:', error);
    } finally {
        // 重置发送时的对话的发送状态
        const sendingChat = chats.find(c => c.id === sendingChatId);
        if (sendingChat) {
            sendingChat.isStreaming = false;
        }
        // 如果当前显示的对话就是发送时的对话，更新UI
        if (currentChatId === sendingChatId) {
            updateUIForCurrentChat();
        }
    }
}

// 根据当前对话状态更新 UI（只影响当前对话）
function updateUIForCurrentChat() {
    const streaming = isCurrentChatStreaming();
    
    if (sendBtn) {
        // 发送按钮：如果当前对话正在发送，或者输入框为空，则禁用
        sendBtn.disabled = streaming || !questionInput.value.trim();
    }
    
    if (questionInput) {
        // 输入框：如果当前对话正在发送，则禁用（只影响当前对话）
        questionInput.disabled = streaming;
        // 更新输入框高度
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    }
}

function createNewChat() {
    // 保存当前对话的输入框内容（如果有）
    saveCurrentChatDraft();
    
    const chatId = `chat_${Date.now()}`;
    const chat = {
        id: chatId,
        title: `新对话 ${++chatCounter}`,
        messages: [],
        createdAt: Date.now(),
        isStreaming: false,  // 每个对话独立的发送状态
        draft: ''  // 每个对话独立的输入框内容
    };
    
    chats.push(chat);
    currentChatId = chatId;
    
    renderChatList();
    loadChat(chatId);
}

function loadChat(chatId) {
    // 保存当前对话的输入框内容（如果有）
    saveCurrentChatDraft();
    
    currentChatId = chatId;
    const chat = chats.find(c => c.id === chatId);
    if (!chat) return;

    // 确保对话对象有必要的属性
    if (chat.isStreaming === undefined) {
        chat.isStreaming = false;
    }
    if (chat.draft === undefined) {
        chat.draft = '';
    }

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

    // 恢复该对话的输入框内容
    if (questionInput && chat.draft !== undefined) {
        questionInput.value = chat.draft || '';
        // 自动调整高度
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    }

    // 根据当前对话状态更新 UI
    updateUIForCurrentChat();
    
    renderChatList();
}

// 保存当前对话的输入框内容
function saveCurrentChatDraft() {
    if (!currentChatId || !questionInput) return;
    const chat = chats.find(c => c.id === currentChatId);
    if (chat) {
        chat.draft = questionInput.value || '';
    }
}

function addMessageToChat(role, content, chatId = null) {
    // 使用指定的 chatId，如果没有指定则使用当前对话ID
    const targetChatId = chatId || currentChatId;
    if (!targetChatId) return;
    
    const targetChat = chats.find(c => c.id === targetChatId);
    if (targetChat) {
        targetChat.messages.push({ role, content });
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

// 显示消息到指定对话（如果该对话当前正在显示，则立即显示；否则只保存到对话记录）
function displayMessageToChat(role, content, chatId) {
    // 先保存到对话记录
    addMessageToChat(role, content, chatId);
    
    // 如果该对话当前正在显示，则立即显示消息
    if (chatId === currentChatId) {
        displayMessage(role, content);
        chatMessages.scrollTop = chatMessages.scrollHeight;
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

// 为指定对话添加流式消息占位符（如果该对话当前正在显示）
function addMessageForStreamingToChat(role, content, chatId) {
    // 如果该对话当前正在显示，则显示占位符
    if (chatId === currentChatId) {
        return addMessageForStreaming(role, content);
    }
    // 如果不在当前显示的对话，创建一个虚拟的占位符（不会显示，但可以用于后续移除）
    const messageDiv = document.createElement('div');
    messageDiv.style.display = 'none';
    messageDiv.className = `message ${role}`;
    messageDiv.setAttribute('data-chat-id', chatId);
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
            successMsg += `<p style="color: blue; text-align: center; font-size: 12px;">📚 向量数据库正在更新中...</p>`;
            progressDiv.innerHTML = successMsg;
            
            // 等待1秒后刷新知识库列表
            setTimeout(async () => {
                await loadKnowledgeBases();
                console.log('知识库列表已刷新');
            }, 1000);
            
            // 清空文件选择
            selectedFiles = [];
            fileInput.value = '';
            document.getElementById('fileInfo').style.display = 'none';
            document.getElementById('confirmUploadBtn').disabled = true;
            
            setTimeout(() => {
                uploadModal.style.display = 'none';
                progressDiv.innerHTML = '';
            }, 3000);
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
        
        let options = '';
        
        if (data.knowledge_bases && data.knowledge_bases.length > 0) {
            // 添加默认知识库
            const defaultKb = data.knowledge_bases.find(kb => kb.name === '');
            if (defaultKb) {
                options += `<option value="${defaultKb.name}">默认知识库 (${defaultKb.file_count} 文件)</option>`;
            }
            
            // 添加其他知识库
            const otherKbs = data.knowledge_bases.filter(kb => kb.name !== '');
            options += otherKbs
                .map(kb => `<option value="${kb.path}">${kb.name} (${kb.file_count} 文件)</option>`)
                .join('');
        }
        
        // 如果没有知识库，添加默认选项
        if (!options) {
            options = '<option value="">默认知识库</option>';
        }
        
        // 只更新知识库选择器
        if (dirSelect) {
            const currentValue = dirSelect.value;
            dirSelect.innerHTML = options;
            // 尝试恢复之前选择的值
            if (currentValue) {
                const option = Array.from(dirSelect.options).find(opt => opt.value === currentValue);
                if (option) {
                    dirSelect.value = currentValue;
                }
            }
            console.log('知识库列表已更新，当前选项数:', dirSelect.options.length);
        }
        
        // 同时更新 upload 知识库
        const uploadDir = data.knowledge_bases?.find(kb => kb.name === 'upload');
        if (uploadDir) {
            console.log('上传目录知识库:', uploadDir);
        }
    } catch (error) {
        console.error('加载知识库失败:', error);
        // 默认选项
        if (dirSelect) {
            dirSelect.innerHTML = '<option value="">默认知识库</option>';
        }
    }
}

