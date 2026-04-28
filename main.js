import * as webllm from "https://esm.run/@mlc-ai/web-llm";

let engine;
let documentText = "";
const selectedModel = "Llama-3-8B-Instruct-v0.1-q4f16_1-MLC"; // 사양에 따라 조정 가능

const messagesContainer = document.getElementById("messages");
const statusLabel = document.getElementById("status-label");
const progressFill = document.getElementById("progress-fill");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// 1. WebLLM 엔진 초기화
async function initEngine() {
    engine = await webllm.CreateEngine(selectedModel, {
        initProgressCallback: (report) => {
            statusLabel.innerText = report.text;
            const progress = report.progress * 100;
            progressFill.style.width = `${progress}%`;
        }
    });
    statusLabel.innerText = "엔진 준비 완료! PDF를 업로드해주세요.";
}

// 2. PDF에서 텍스트 추출 (pdf.js 사용)
async function extractText(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let fullText = "";
    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        fullText += content.items.map(item => item.str).join(" ") + "\n";
    }
    return fullText;
}

// 3. 파일 업로드 이벤트
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

dropZone.onclick = () => fileInput.click();
fileInput.onchange = async (e) => {
    const file = e.target.files[0];
    if (file) {
        statusLabel.innerText = "문서 읽는 중...";
        documentText = await extractText(file);
        statusLabel.innerText = "학습 완료! 질문을 입력하세요.";
        userInput.disabled = false;
        sendBtn.disabled = false;
    }
};

// 4. 질문 답변 로직
async function askQuestion() {
    const text = userInput.value.strip();
    if (!text) return;

    appendMessage("user", text);
    userInput.value = "";

    const prompt = [
        { role: "system", content: "너는 문서 보조원이야. 반드시 아래 제공된 문맥만을 사용하여 한국어로 답해줘. 문맥에 없는 내용은 모른다고 답해.\n\n문맥: " + documentText },
        { role: "user", content: text }
    ];

    const reply = await engine.chat.completions.create({ messages: prompt });
    appendMessage("assistant", reply.choices[0].message.content);
}

function appendMessage(role, content) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;
    msgDiv.innerText = content;
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

sendBtn.onclick = askQuestion;
initEngine();