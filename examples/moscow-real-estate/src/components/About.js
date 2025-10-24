import React from 'react';
import { motion } from 'framer-motion';
import { FaAward, FaUsers, FaHandshake } from 'react-icons/fa';
import './About.css';

const About = () => {
  const stats = [
    {
      icon: FaHandshake,
      number: 300,
      suffix: '+',
      label: 'Успешных сделок',
      color: '#1e3a8a'
    },
    {
      icon: FaAward,
      number: 12,
      suffix: '',
      label: 'Лет опыта',
      color: '#3b82f6'
    },
    {
      icon: FaUsers,
      number: 95,
      suffix: '%',
      label: 'Довольных клиентов',
      color: '#60a5fa'
    }
  ];

  return (
    <section id="about" className="about">
      <div className="container">
        <motion.div
          className="about-content"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="about-text">
            <div className="about-badge">
              <span>🏆 О компании</span>
            </div>
            
            <h2 className="about-title">
              <span className="gradient-text">Moscow Elite Properties</span>
              <br />
              Ваш надежный партнер в мире премиальной недвижимости
            </h2>
            
            <div className="about-description">
              <p>
                Мы - ведущее агентство недвижимости в Москве с более чем 12-летним опытом работы на рынке. 
                Наша команда профессионалов поможет вам найти идеальную недвижимость или выгодно продать имеющуюся.
              </p>
              <p>
                Мы специализируемся на премиальном сегменте недвижимости и работаем с самыми требовательными клиентами. 
                Наша репутация основана на честности, профессионализме и результативности.
              </p>
            </div>
          </div>

          <div className="about-visual">
            <div className="about-image-container">
              <img
                src="https://images.unsplash.com/photo-1560472354-b33ff0c44a43?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
                alt="Moscow Office"
                className="about-image"
              />
            </div>

            <div className="stats-container">
              {stats.map((stat, index) => (
                <motion.div
                  key={index}
                  className="stat-card"
                  style={{ backgroundColor: stat.color }}
                  whileHover={{ scale: 1.05 }}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <div className="stat-icon">
                    <stat.icon />
                  </div>
                  <div className="stat-number">
                    {stat.number}{stat.suffix}
                  </div>
                  <div className="stat-label">{stat.label}</div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default About;
