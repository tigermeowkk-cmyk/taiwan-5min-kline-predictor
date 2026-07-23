# Supabase 外部儲存設定步驟（準確率紀錄用）

等有空要接的時候，照下面順序做就好。

## 1. 建 Project（在你已有的 Organization 底下）
- 進 Supabase → 你的 Organization → **New Project**
- 專案名稱隨意，例如 `taiwan-stock-app`
- Database Password：設一組並記下來（等下連線字串要用）
- Region：Singapore（或離台灣近的）
- Plan：Free
- 下面 Enable Data API / Automatically expose new tables / Enable automatic RLS：**都不用勾**（我們不走 Supabase 的 REST API，直接用連線字串連 Postgres）

## 2. 拿連線字串
- 進到剛建好的 Project 首頁
- 右上角點 **Connect** 按鈕
- 跳出的視窗裡切到 **Transaction pooler** 分頁（不要用 Direct connection，那個是 IPv6，Render 連不上）
- 複製那串 `postgresql://postgres.xxxx:[YOUR-PASSWORD]@aws-0-xxxx.pooler.supabase.com:6543/postgres`
- 把 `[YOUR-PASSWORD]` 換成第1步設的資料庫密碼（整段換掉，不留中括號；密碼如果有 `@ : / #` 之類符號要跟我說，需要做 URL encode）

## 3. 設定到 Render
- Render 後台 → 這個 Streamlit 服務 → **Environment**
- 新增一筆：
  - Key: `POSTGRESQL_URL`
  - Value: 上面換好密碼的完整連線字串
- 存檔，Render 會自動重新部署

## 4. 驗證
- 部署完，打開 app 跑一次預測
- 展開「📊 歷史準確率紀錄」
  - 綠色 ✅「目前寫入 Supabase 資料庫」= 成功
  - 黃色警告 = `POSTGRESQL_URL` 沒抓到，回頭檢查 Render 環境變數
- 表格 (`accuracy_log`) 程式會自動建立，不用手動去 Supabase 建
