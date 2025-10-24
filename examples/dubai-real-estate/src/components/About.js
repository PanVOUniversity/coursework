import React, { useState, useEffect } from 'react';
import { motion, useAnimation } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { FaAward, FaUsers, FaHandshake, FaGlobe } from 'react-icons/fa';
import './About.css';

const About = () => {
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const [counts, setCounts] = useState({
    deals: 0,
    years: 0,
    clients: 0,
    properties: 0
  });

  const controls = useAnimation();

  useEffect(() => {
    if (inView) {
      controls.start('visible');
      
      // Animate counters
      const targets = { deals: 500, years: 15, clients: 98, properties: 1000 };
      const duration = 2000;
      const steps = 60;
      const stepDuration = duration / steps;

      let step = 0;
      const timer = setInterval(() => {
        step++;
        const progress = step / steps;
        
        setCounts({
          deals: Math.floor(targets.deals * progress),
          years: Math.floor(targets.years * progress),
          clients: Math.floor(targets.clients * progress),
          properties: Math.floor(targets.properties * progress)
        });

        if (step >= steps) {
          clearInterval(timer);
          setCounts(targets);
        }
      }, stepDuration);
    }
  }, [inView, controls]);

  const stats = [
    {
      icon: FaHandshake,
      number: counts.deals,
      suffix: '+',
      label: 'Успешных сделок',
      color: '#2d5a27'
    },
    {
      icon: FaAward,
      number: counts.years,
      suffix: '',
      label: 'Лет опыта',
      color: '#4a7c59'
    },
    {
      icon: FaUsers,
      number: counts.clients,
      suffix: '%',
      label: 'Довольных клиентов',
      color: '#6b8e6b'
    },
    {
      icon: FaGlobe,
      number: counts.properties,
      suffix: '+',
      label: 'Объектов в портфеле',
      color: '#a8d5a8'
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  return (
    <section id="about" className="about">
      <div className="container">
        <motion.div
          className="about-content"
          ref={ref}
          variants={containerVariants}
          initial="hidden"
          animate={controls}
        >
          <motion.div className="about-text" variants={itemVariants}>
            <div className="about-badge">
              <span>🏆 О компании</span>
            </div>
            
            <h2 className="about-title">
              <span className="gradient-text">Dubai Elite Properties</span>
              <br />
              Ваш надежный партнер в мире премиальной недвижимости
            </h2>
            
            <div className="about-description">
              <p>
                Мы - ведущее агентство недвижимости в Дубае с более чем 15-летним опытом работы на рынке. 
                Наша команда профессионалов поможет вам найти идеальную недвижимость или выгодно продать имеющуюся.
              </p>
              <p>
                Мы специализируемся на премиальном сегменте недвижимости и работаем с самыми требовательными клиентами. 
                Наша репутация основана на честности, профессионализме и результативности.
              </p>
            </div>

            <div className="about-features">
              <div className="feature-item">
                <div className="feature-icon">
                  <FaAward />
                </div>
                <div className="feature-content">
                  <h4>Премиальное качество</h4>
                  <p>Работаем только с лучшими объектами класса люкс</p>
                </div>
              </div>
              
              <div className="feature-item">
                <div className="feature-icon">
                  <FaHandshake />
                </div>
                <div className="feature-content">
                  <h4>Персональный подход</h4>
                  <p>Индивидуальная работа с каждым клиентом</p>
                </div>
              </div>
              
              <div className="feature-item">
                <div className="feature-icon">
                  <FaGlobe />
                </div>
                <div className="feature-content">
                  <h4>Международный опыт</h4>
                  <p>Обслуживание клиентов со всего мира</p>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div className="about-visual" variants={itemVariants}>
            <div className="about-image-container">
              <img
                src="https://images.unsplash.com/photo-1560472354-b33ff0c44a43?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
                alt="Dubai Office"
                className="about-image"
              />
              <div className="image-overlay" />
            </div>

            <div className="stats-container">
              {stats.map((stat, index) => (
                <motion.div
                  key={index}
                  className="stat-card"
                  style={{ backgroundColor: stat.color }}
                  whileHover={{ scale: 1.05 }}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
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
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default About;
