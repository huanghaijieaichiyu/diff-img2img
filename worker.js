/**
 * Cloudflare Worker: Personal Dashboard Pro
 * 修复了缩进、变量转义以及新闻抓取的健壮性
 */

export default {
  async fetch(request) {
    const url = new URL(request.url);

    // === 1. 后端 API: 获取百度热搜 (解决跨域问题) ===
    if (url.pathname === '/api/news') {
      return await handleNewsRequest();
    }

    // === 2. 前端页面渲染 ===
    // 获取请求者的地理位置信息 (Cloudflare 注入)
    const cf = request.cf || {};
    const locationData = {
      city: cf.city || 'Beijing',
      lat: cf.latitude || '39.9042',
      lon: cf.longitude || '116.4074'
    };

    // 返回 HTML 响应
    return new Response(generateHTML(locationData), {
      headers: { 'content-type': 'text/html;charset=UTF-8' },
    });
  },
};

/**
 * 逻辑函数：抓取百度热搜
 */
async function handleNewsRequest() {
  try {
    const targetUrl = 'https://top.baidu.com/board?tab=realtime';
    const response = await fetch(targetUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; Cloudflare-Worker/1.0)'
      }
    });

    if (!response.ok) throw new Error('Network error');
    const html = await response.text();

    // 正则提取热搜标题
    // 目标结构: <div class="c-single-text-ellipsis"> 标题 </div>
    const items = [];
    const regex = /<div class="c-single-text-ellipsis">\s*(.*?)\s*<\/div>/g;
    let match;
    let count = 0;

    while ((match = regex.exec(html)) !== null && count < 15) {
      const title = match[1].trim();
      // 过滤掉无效数据（注释、链接等）
      if (title && !title.includes('<')) {
        items.push(title);
        count++;
      }
    }

    return new Response(JSON.stringify({ success: true, data: items }), {
      headers: { 'content-type': 'application/json;charset=UTF-8' },
    });
  } catch (e) {
    return new Response(JSON.stringify({ success: false, message: e.message }), {
      headers: { 'content-type': 'application/json;charset=UTF-8' },
      status: 500
    });
  }
}

function generateHTML(loc) {
return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Personal Dashboard Pro</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
  <style>
    /* 简化的样式 */
    :root {
      --bg: #0f1724;
      --bg-glass: rgba(255,255,255,0.03);
      --border-glass: rgba(255,255,255,0.06);
    }
    html,body{height:100%; margin:0; font-family:Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;}
    .app-wrap{min-height:100%; display:flex; flex-direction:column; background:linear-gradient(180deg,#071023,#0f1724); color:#fff;}
    .topbar{height:64px; display:flex; align-items:center; padding:0 20px; border-bottom:1px solid rgba(255,255,255,0.03);}
    .sidebar{width:72px; padding:20px 0; background:transparent; border-right:1px solid rgba(255,255,255,0.02);}
    .nav-item{padding:12px; text-align:center; cursor:pointer; color:#cbd5e1; margin-bottom:8px;}
    .nav-item.active{background:rgba(255,255,255,0.04); border-radius:12px; color:#fff;}
    .spacer{flex:1;}
    .main-content{flex:1; padding:24px; display:flex; gap:20px; flex-direction:column;}
    .widget-grid{display:flex; gap:20px; align-items:flex-start;}
    .glass-card{background:var(--bg-glass); padding:18px; border-radius:14px; min-width:220px; border:1px solid var(--border-glass);}
    .time-widget{width:260px;}
    .clock-text{font-size:2.6rem; font-weight:600;}
    .date-text{font-size:0.95rem; color:#bcd;}
    .news-widget{flex:1;}
    .news-list{list-style:none; padding:0; margin:0;}
    .news-item{padding:8px 10px; display:flex; gap:8px; align-items:center; border-radius:8px; cursor:pointer;}
    .news-item:hover{background:rgba(255,255,255,0.02);}
    .news-index{width:28px; text-align:center; color:#ffb86b;}
    .search-container{margin-top:10px;}
    .search-wrapper{display:flex; align-items:center; gap:10px; max-width:820px;}
    .engine-switch{display:flex; align-items:center; gap:10px; cursor:pointer; padding:8px; border-radius:10px; background:rgba(255,255,255,0.03);}
    .engine-dropdown{display:none; position:absolute; background:#0b1220; padding:8px; border-radius:8px; margin-top:8px;}
    .engine-dropdown.show{display:block;}
    .engine-option{padding:8px; cursor:pointer; color:#cbd5e1;}
    .app-grid{display:flex; gap:12px; flex-wrap:wrap;}
    .app-item{display:flex; flex-direction:column; align-items:center; width:84px; text-decoration:none; color:inherit;}
    .app-icon{width:64px; height:64px; border-radius:12px; display:flex; align-items:center; justify-content:center; margin-bottom:8px; background:linear-gradient(135deg,#2b6cb0,#4c51bf);}
    .fab{position:fixed; right:28px; bottom:28px; width:56px; height:56px; border-radius:50%; background:#111827; display:flex; align-items:center; justify-content:center; cursor:pointer; border:1px solid rgba(255,255,255,0.04);}
    .modal-overlay{display:none; position:fixed; inset:0; align-items:center; justify-content:center; background:rgba(0,0,0,0.6);}
    .modal-box{background:#071023; padding:20px; border-radius:12px; width:480px;}
    .btn{padding:8px 12px; border-radius:8px; cursor:pointer; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.03); color:#fff;}
    .btn-save{background:#2563eb;}
  </style>
</head>
<body>
  <div class="app-wrap">
    <div class="topbar">
      <div style="font-weight:600; font-size:1.05rem;">Personal Dashboard Pro</div>
      <div style="margin-left:auto; color:#9ca3af;">Location: ${loc.city}</div>
    </div>

    <div style="display:flex; flex:1; min-height:calc(100vh - 64px);">
    <nav class="sidebar">
      <div class="nav-item active" data-target="home" title="主页"><i class="fa-solid fa-house"></i></div>
      <div class="nav-item" data-target="memo" title="便签"><i class="fa-solid fa-feather"></i></div>
      <div class="spacer"></div>
      <div class="nav-item" data-target="settings" title="设置"><i class="fa-solid fa-gear"></i></div>
    </nav>

    <main class="main-content">
      <section id="home" class="tab-section active">
        <div class="widget-grid">
          <div class="glass-card time-widget">
            <div class="clock-text" id="clock">--:--</div>
            <div class="date-text" id="date">加载中...</div>
            <div class="weather-row">
              <span id="weather-icon"><i class="fa-solid fa-spin fa-spinner"></i></span>
              <span style="margin: 0 8px">${loc.city}</span>
              <span id="weather-temp">--</span>
            </div>
          </div>
          <div class="glass-card news-widget">
            <div class="news-header"><i class="fa-brands fa-hotjar"></i> 百度热搜</div>
            <ul class="news-list" id="news-list">
              <li style="text-align:center; padding:20px; color:#888;">加载中...</li>
            </ul>
          </div>
        </div>

        <div class="search-container">
          <div class="search-wrapper">
            <div class="engine-switch" id="engine-btn">
              <i class="fa-brands fa-baidu" id="curr-engine-icon" style="font-size:1.4rem; color:#fff;"></i>
              <div class="engine-dropdown" id="engine-menu"></div>
            </div>
            <input type="text" id="search-input" placeholder="Search..." autocomplete="off">
          </div>
        </div>

        <div class="app-grid" id="app-grid"></div>
        <div class="fab" onclick="App.openModal()"><i class="fa-solid fa-plus"></i></div>
      </section>

      <section id="memo" class="tab-section">
        <h2 style="margin-bottom:20px; font-weight:300;">随手记</h2>
        <div style="background:var(--bg-glass); padding:20px; border-radius:20px; height:70vh; border:1px solid var(--border-glass);">
          <textarea id="memo-area" style="width:100%; height:100%; background:transparent; border:none; color:#fff; font-size:1.2rem; resize:none; outline:none;" placeholder="记录你的想法..."></textarea>
        </div>
      </section>

      <section id="settings" class="tab-section">
        <div style="max-width:600px; margin:0 auto;">
          <h2 style="margin-bottom:30px;">设置</h2>
          <div class="glass-card" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
            <span>自定义壁纸 URL</span>
            <input type="text" id="bg-input" placeholder="https://..." style="background:rgba(0,0,0,0.3); border:1px solid #555; color:#fff; padding:8px; border-radius:6px; width:250px;">
          </div>
          <div class="glass-card" style="display:flex; justify-content:space-between; align-items:center;">
            <span>清除数据</span>
            <button class="btn" style="background:#d63031; color:#fff;" onclick="App.resetData()">重置所有</button>
          </div>
          <div style="margin-top:20px; text-align:right;">
            <button class="btn btn-save" onclick="App.saveSettings()">保存应用</button>
          </div>
        </div>
      </section>
    </main>
  </div>

  <div id="app-modal" class="modal-overlay">
    <div class="modal-box">
      <h3 style="margin-bottom:20px;">添加图标</h3>
      <div class="input-group"><input type="text" id="new-name" placeholder="名称"></div>
      <div class="input-group"><input type="text" id="new-url" placeholder="网址 (https://...)"></div>
      <div class="input-group"><input type="text" id="new-icon" placeholder="图标代码 (如: fa-brands fa-github)"></div>
      <div class="modal-actions">
        <button class="btn" style="background:transparent; color:#aaa;" onclick="App.closeModal()">取消</button>
        <button class="btn btn-save" onclick="App.addApp()">确定</button>
      </div>
    </div>
  </div>

  <script>
    // 注入服务端变量
    const CF_COORDS = { lat: "${loc.lat}", lon: "${loc.lon}" };

    const ENGINES = {
      baidu: { name: '百度', icon: 'fa-brands fa-baidu', url: 'https://www.baidu.com/s?wd=' },
      bing: { name: '必应', icon: 'fa-brands fa-microsoft', url: 'https://www.bing.com/search?q=' },
      google: { name: '谷歌', icon: 'fa-brands fa-google', url: 'https://www.google.com/search?q=' },
      ai: { name: 'AI', icon: 'fa-solid fa-robot', url: 'https://www.perplexity.ai/search?q=' }
    };

    const DEFAULT_APPS = [
      { name: "百度", url: "https://baidu.com", icon: "fa-brands fa-baidu", color: "grad-blue" },
      { name: "Bilibili", url: "https://bilibili.com", icon: "fa-brands fa-bilibili", color: "grad-pink" },
      { name: "微博", url: "https://weibo.com", icon: "fa-brands fa-weibo", color: "grad-red" },
      { name: "知乎", url: "https://zhihu.com", icon: "fa-solid fa-book-open", color: "grad-blue" },
      { name: "GitHub", url: "https://github.com", icon: "fa-brands fa-github", color: "grad-dark" },
      { name: "ChatGPT", url: "https://chat.openai.com", icon: "fa-solid fa-robot", color: "grad-green" }
    ];

    const App = {
      data: { apps: [], engine: 'baidu', settings: { bg: '' } },

      init() {
        this.loadData();
        this.renderUI();
        this.startClock();
        this.fetchWeather();
        this.fetchNews();
        this.bindEvents();
      },

      loadData() {
        this.data.apps = JSON.parse(localStorage.getItem('d_apps')) || DEFAULT_APPS;
        this.data.engine = localStorage.getItem('d_engine') || 'baidu';
        this.data.settings = JSON.parse(localStorage.getItem('d_settings')) || { bg: '' };
        
        // 恢复便签
        document.getElementById('memo-area').value = localStorage.getItem('d_memo') || '';
        
        // 恢复壁纸
        if(this.data.settings.bg) {
          document.getElementById('bg-layer').style.backgroundImage = \`url('\${this.data.settings.bg}')\`;
          document.getElementById('bg-input').value = this.data.settings.bg;
        }
      },

      renderUI() {
        // 1. 渲染图标
        const grid = document.getElementById('app-grid');
        grid.innerHTML = '';
        this.data.apps.forEach((app, idx) => {
          const a = document.createElement('a');
          a.className = 'app-item';
          a.href = app.url;
          a.target = '_blank';
          a.innerHTML = \`
            <div class="app-icon \${app.color || 'grad-blue'}"><i class="\${app.icon}"></i></div>
            <div class="app-name">\${app.name}</div>
          \`;
          a.oncontextmenu = (e) => {
            e.preventDefault();
            if(confirm('删除此图标?')) {
              this.data.apps.splice(idx, 1);
              localStorage.setItem('d_apps', JSON.stringify(this.data.apps));
              this.renderUI();
            }
          };
          grid.appendChild(a);
        });

        // 2. 渲染搜索引擎
        this.updateSearchIcon();
        const menu = document.getElementById('engine-menu');
        menu.innerHTML = '';
        Object.keys(ENGINES).forEach(key => {
          const div = document.createElement('div');
          div.className = 'engine-option';
          div.innerHTML = \`<i class="\${ENGINES[key].icon}"></i> \${ENGINES[key].name}\`;
          div.onclick = (e) => {
            e.stopPropagation();
            this.data.engine = key;
            localStorage.setItem('d_engine', key);
            this.updateSearchIcon();
            menu.classList.remove('show');
          };
          menu.appendChild(div);
        });
      },

      updateSearchIcon() {
        const cur = ENGINES[this.data.engine];
        const icon = document.getElementById('curr-engine-icon');
        icon.className = \`\${cur.icon}\`;
        document.getElementById('search-input').placeholder = \`在 \${cur.name} 中搜索...\`;
      },

      startClock() {
        const update = () => {
          const now = new Date();
          document.getElementById('clock').innerText = now.toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit'});
          document.getElementById('date').innerText = now.toLocaleDateString('zh-CN', {month:'long', day:'numeric', weekday:'long'});
        };
        setInterval(update, 1000);
        update();
      },

      async fetchWeather() {
        try {
          const res = await fetch(\`https://api.open-meteo.com/v1/forecast?latitude=\${CF_COORDS.lat}&longitude=\${CF_COORDS.lon}&current_weather=true\`);
          const data = await res.json();
          const code = data.current_weather.weathercode;
          const temp = Math.round(data.current_weather.temperature);
          
          let icon = 'fa-cloud';
          if(code === 0) icon = 'fa-sun';
          else if(code <= 3) icon = 'fa-cloud-sun';
          else if(code >= 95) icon = 'fa-bolt';
          else if(code >= 60) icon = 'fa-cloud-rain';

          document.getElementById('weather-temp').innerText = \`\${temp}°C\`;
          document.getElementById('weather-icon').innerHTML = \`<i class="fa-solid \${icon}"></i>\`;
        } catch(e) { console.error(e); }
      },

      async fetchNews() {
        const list = document.getElementById('news-list');
        try {
          const res = await fetch('/api/news');
          const json = await res.json();
          if(json.success && json.data.length) {
            list.innerHTML = '';
            json.data.forEach((title, idx) => {
              const li = document.createElement('li');
              li.className = 'news-item';
              li.innerHTML = \`<span class="news-index">\${idx+1}</span> <span class="news-title">\${title}</span>\`;
              li.onclick = () => window.open('https://www.baidu.com/s?wd=' + encodeURIComponent(title));
              list.appendChild(li);
            });
          }
        } catch(e) { list.innerHTML = '<li style="text-align:center; padding:20px;">暂无数据</li>'; }
      },

      bindEvents() {
        // Tab 切换
        document.querySelectorAll('.nav-item').forEach(el => {
          el.addEventListener('click', () => {
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.querySelectorAll('.tab-section').forEach(t => t.classList.remove('active'));
            el.classList.add('active');
            document.getElementById(el.dataset.target).classList.add('active');
          });
        });

        // 搜索框
        const input = document.getElementById('search-input');
        input.addEventListener('keypress', (e) => {
          if(e.key === 'Enter' && input.value.trim()) {
            const q = input.value.trim();
            const url = q.startsWith('http') ? q : ENGINES[this.data.engine].url + encodeURIComponent(q);
            window.open(url, '_blank');
            input.value = '';
          }
        });

        // 引擎菜单
        document.getElementById('engine-btn').onclick = (e) => {
          e.stopPropagation();
          document.getElementById('engine-menu').classList.toggle('show');
        };
        window.onclick = () => document.getElementById('engine-menu').classList.remove('show');

        // 便签保存
        document.getElementById('memo-area').addEventListener('input', (e) => {
          localStorage.setItem('d_memo', e.target.value);
        });
      },

      // Modal Actions
      openModal() { document.getElementById('app-modal').style.display = 'flex'; },
      closeModal() { document.getElementById('app-modal').style.display = 'none'; },
      
      addApp() {
        const name = document.getElementById('new-name').value;
        const url = document.getElementById('new-url').value;
        const icon = document.getElementById('new-icon').value || 'fa-solid fa-link';
        if(name && url) {
          const colors = ['grad-blue', 'grad-red', 'grad-green', 'grad-purple', 'grad-pink'];
          this.data.apps.push({ 
            name, url, icon, 
            color: colors[Math.floor(Math.random()*colors.length)] 
          });
          localStorage.setItem('d_apps', JSON.stringify(this.data.apps));
          this.renderUI();
          this.closeModal();
          // 清空输入
          document.getElementById('new-name').value = '';
          document.getElementById('new-url').value = '';
        }
      },

      saveSettings() {
        const bg = document.getElementById('bg-input').value;
        this.data.settings.bg = bg;
        localStorage.setItem('d_settings', JSON.stringify(this.data.settings));
        if(bg) document.getElementById('bg-layer').style.backgroundImage = \`url('\${bg}')\`;
        alert('设置已保存');
      },

      resetData() {
        if(confirm('确定重置所有数据？')) {
          localStorage.clear();
          location.reload();
        }
      }
    };

    document.addEventListener('DOMContentLoaded', () => App.init());
  </script>
</body>
</html>`;
}