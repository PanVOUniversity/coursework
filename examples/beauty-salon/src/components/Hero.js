import React from 'react';
import { motion } from 'framer-motion';
import { FaArrowDown, FaStar } from 'react-icons/fa';
import './styles.css';

const Hero = () => {
  return (
    <section id="home" className="hero">
      <div className="hero-background">
        <img src="https://images.unsplash.com/photo-1560066984-138dadb4c035?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80" alt="Beauty Salon" className="hero-image" />
        <div className="hero-overlay" />
      </div>

      <div className="container">
        <motion.div className="hero-content" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8 }}>
          <motion.div className="hero-badge" initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}>
            <span>💇‍♀️ Премиальная парикмахерская</span>
          </motion.div>

          <motion.h1 className="hero-title" initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.4 }}>
            <span className="gradient-text">Красота</span> начинается здесь
          </motion.h1>

          <motion.p className="hero-subtitle" initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.6 }}>
            Профессиональные стрижки, окрашивание и уход за волосами от лучших мастеров
          </motion.p>

          <motion.div className="hero-buttons" initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.8 }}>
            <motion.button className="btn btn-primary" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              Записаться онлайн
            </motion.button>
          </motion.div>

          <motion.div className="hero-stats" initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 1 }}>
            <div className="stat-item">
              <FaStar className="stat-icon" />
              <div className="stat-number">5.0</div>
              <div className="stat-label">Рейтинг</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">1000+</div>
              <div className="stat-label">Клиентов</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">8</div>
              <div className="stat-label">Лет опыта</div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default Hero;
