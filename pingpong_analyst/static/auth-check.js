/**
 * Auth — 全局认证检查 + fetch 拦截器
 * 在所有页面中引入此文件，自动处理 token 注入和未授权跳转。
 */

(function () {
  const TOKEN_KEY = "pp_token";
  const USER_KEY = "pp_user";

  function getToken() {
    return localStorage.getItem(TOKEN_KEY);
  }

  function clearAuth() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  }

  // ---------- fetch 拦截器：自动注入 Authorization 头 ----------
  const originalFetch = window.fetch;
  window.fetch = function (input, init) {
    init = init || {};
    const url = typeof input === "string" ? input : input?.url || "";
    // 只对 /api/ 请求注入 token（排除 login 和 health）
    if (url.startsWith("/api/") && !url.includes("/api/auth/login") && !url.includes("/api/health")) {
      const token = getToken();
      if (token) {
        init.headers = new Headers(init.headers || {});
        init.headers.set("Authorization", `Bearer ${token}`);
      }
    }
    return originalFetch.call(this, input, init).then((res) => {
      // 收到 401 时清除认证并跳转登录
      if (res.status === 401 && !window.location.pathname.startsWith("/login")) {
        clearAuth();
        window.location.href = "/login";
      }
      return res;
    });
  };

  // ---------- XMLHttpRequest 拦截器 ----------
  const originalOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this._url = url;
    return originalOpen.call(this, method, url, ...rest);
  };
  const originalSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.send = function (body) {
    if (this._url && this._url.startsWith("/api/") && !this._url.includes("/api/auth/login")) {
      const token = getToken();
      if (token) {
        this.setRequestHeader("Authorization", `Bearer ${token}`);
      }
    }
    return originalSend.call(this, body);
  };

  // ---------- 页面加载时检查认证 ----------
  // 排除登录页本身
  if (!window.location.pathname.startsWith("/login")) {
    const token = getToken();
    if (!token) {
      window.location.href = "/login";
    } else {
      // 异步验证 token 有效性
      fetch("/api/auth/me").then((res) => {
        if (!res.ok) {
          clearAuth();
          window.location.href = "/login";
        }
      }).catch(() => {
        clearAuth();
        window.location.href = "/login";
      });
    }
  }

  // ---------- 暴露工具函数 ----------
  window.PPAuth = {
    getToken,
    getUser: () => {
      try { return JSON.parse(localStorage.getItem(USER_KEY) || "null"); }
      catch { return null; }
    },
    clearAuth,
    logout: async () => {
      clearAuth();
      window.location.href = "/login";
    },
    // 给 <video src>/<img src>/<a href> 等 URL 附加 token query param
    authUrl: (url) => {
      const token = getToken();
      if (!token || !url || !url.startsWith("/api/")) return url;
      const sep = url.includes("?") ? "&" : "?";
      return `${url}${sep}token=${encodeURIComponent(token)}`;
    },
  };
})();
