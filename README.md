# 🚀 Три React лендинга - Готовые проекты

Три современных одностраничных лендинга на React с красивым дизайном и анимациями.

## 📦 Проекты

### 1. 🏢 Moscow Elite Properties (moscow-real-estate/)
**Агентство недвижимости в Москве**
- **Цветовая схема**: Синяя (#1e3a8a)
- **Особенности**: Карточки недвижимости, интерактивная статистика, контактная форма
- **Секции**: Hero, Features, Properties, Services, About, Contact, Footer

### 2. 🎨 Creative Agency (marketing-agency/)
**Маркетинговое агентство**
- **Цветовая схема**: Фиолетовая (#7c3aed)
- **Особенности**: Портфолио работ, услуги, анимированная галерея
- **Секции**: Hero, Services, Portfolio, About, Contact, Footer

### 3. 💇‍♀️ Beauty Salon (beauty-salon/)
**Премиальная парикмахерская**
- **Цветовая схема**: Розовая (#ec4899)
- **Особенности**: Галерея работ, прайс-лист, онлайн-запись
- **Секции**: Hero, Services, Gallery, Pricing, Contact, Footer

## 🛠 Технологии

- **React 18** - Основной фреймворк
- **Framer Motion** - Плавные анимации
- **React Icons** - Иконки
- **React Hot Toast** - Уведомления
- **CSS3** - Адаптивный дизайн

## 🚀 Запуск проектов

### Способ 1: Запуск каждого проекта отдельно

```bash
# 1. Агентство недвижимости в Москве
cd moscow-real-estate
npm install
npm start
# Откроется на http://localhost:3000

# 2. Маркетинговое агентство
cd marketing-agency
npm install
npm start
# Откроется на http://localhost:3000

# 3. Парикмахерская
cd beauty-salon
npm install
npm start
# Откроется на http://localhost:3000
```

### Способ 2: Запуск всех проектов одновременно

```bash
# Установка зависимостей для всех проектов
cd moscow-real-estate && npm install && cd ..
cd marketing-agency && npm install && cd ..
cd beauty-salon && npm install && cd ..

# Запуск на разных портах (используйте разные терминалы)
# Терминал 1:
cd moscow-real-estate && PORT=3001 npm start

# Терминал 2:
cd marketing-agency && PORT=3002 npm start

# Терминал 3:
cd beauty-salon && PORT=3003 npm start
```

## ✨ Особенности дизайна

### Общие возможности:
- ✅ Полностью адаптивный дизайн (Desktop, Tablet, Mobile)
- ✅ Плавные анимации при скролле
- ✅ Интерактивные элементы с hover-эффектами
- ✅ Мобильное меню с анимацией
- ✅ Градиентные фоны и элементы
- ✅ Контактные формы с валидацией
- ✅ Оптимизированные изображения из Unsplash

### Анимации:
- Fade In / Fade Out
- Slide In (Left, Right, Up, Down)
- Scale animations
- Stagger children animations
- Hover effects
- Smooth scroll

## 📱 Адаптивность

Все сайты полностью адаптированы для:
- **Desktop** (1200px+)
- **Tablet** (768px - 1199px)
- **Mobile** (до 767px)

## 🎨 Цветовые схемы

### Moscow Elite Properties
```css
--primary-blue: #1e3a8a
--secondary-blue: #3b82f6
--accent-blue: #60a5fa
```

### Creative Agency
```css
--primary-purple: #7c3aed
--secondary-purple: #a855f7
--accent-purple: #c084fc
```

### Beauty Salon
```css
--primary-pink: #ec4899
--secondary-pink: #f472b6
--accent-pink: #fbcfe8
```

## 📂 Структура проекта

```
project/
├── moscow-real-estate/          # Агентство недвижимости
│   ├── public/
│   ├── src/
│   │   ├── components/          # React компоненты
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
│
├── marketing-agency/            # Маркетинговое агентство
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
│
└── beauty-salon/                # Парикмахерская
    ├── public/
    ├── src/
    │   ├── components/
    │   ├── App.js
    │   └── index.js
    └── package.json
```

## 🔧 Build для продакшена

```bash
# Для каждого проекта
cd [project-name]
npm run build
```

Создастся папка `build` с оптимизированными файлами.

## 📝 Примечания

- Все изображения взяты из Unsplash
- Формы отправляют данные через toast-уведомления (можно интегрировать с backend)
- Проекты готовы к деплою на Netlify, Vercel или другие платформы

## 🎯 Готово к использованию!

Все три сайта полностью функциональны и готовы к запуску. Просто выберите нужный проект и следуйте инструкциям выше.

---

**Созданo с ❤️ используя React и Framer Motion**
