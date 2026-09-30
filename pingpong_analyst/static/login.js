/**
 * Login — 登录页逻辑（含验证码）
 */

const $ = (id) => document.getElementById(id);
let currentCaptchaId = "";

// ---------- 验证码 ----------
async function loadCaptcha() {
  try {
    const res = await fetch("/api/auth/captcha");
    const data = await res.json();
    currentCaptchaId = data.captcha_id;
    $("captchaImg").src = data.image;
  } catch {
    $("captchaImg").alt = "验证码加载失败";
  }
}

$("captchaImg").addEventListener("click", loadCaptcha);

// ---------- 已登录检查 ----------
async function checkExistingAuth() {
  const token = localStorage.getItem("pp_token");
  if (token) {
    try {
      const res = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        window.location.href = "/";
        return;
      }
    } catch {}
    localStorage.removeItem("pp_token");
  }
  // 检查是否首次启动（无用户），决定是否显示默认密码提示
  try {
    const statusRes = await fetch("/api/auth/status");
    const statusData = await statusRes.json();
    if (statusData.needs_init) {
      const hint = $("loginHint");
      hint.style.display = "block";
      hint.textContent = "首次启动 · 默认管理员: admin / admin123（请尽快修改密码）";
    }
  } catch {}
  loadCaptcha();
}

// ---------- 登录提交 ----------
$("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("loginBtn");
  const errorEl = $("loginError");
  const username = $("username").value.trim();
  const password = $("password").value;
  const captchaText = $("captcha").value.trim();

  btn.disabled = true;
  btn.textContent = "登录中...";
  errorEl.hidden = true;

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username,
        password,
        captcha_id: currentCaptchaId,
        captcha_text: captchaText,
      }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "登录失败");
    }

    localStorage.setItem("pp_token", data.token);
    localStorage.setItem("pp_user", JSON.stringify(data.user));
    window.location.href = "/";
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
    btn.disabled = false;
    btn.textContent = "登录";
    $("captcha").value = "";
    loadCaptcha();
  }
});

checkExistingAuth();
