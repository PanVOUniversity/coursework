import React, { useEffect, useState } from 'react';
import { motion, useAnimation } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { FaArrowDown, FaPlay, FaLightbulb, FaChartLine, FaUsers } from 'react-icons/fa';
import './Hero.css';

const Hero = () => {
  const [currentImage, setCurrentImage] = useState(0);
  const controls = useAnimation();
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: false
  });

  const images = [
    'https://images.unsplash.com/photo-1552664730-d307ca884978?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80',
    'https://images.unsplash.com/photo-1559136555-9303baea8ebd?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80',
    'https://images.unsplash.com/photo-1553877522-43269d4ea984?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80'
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentImage((prev) => (prev + 1) % images.length);
    }, 5000);
    return () => clearInterval(interval);
  }, [images.length]);

  useEffect(() => {
    if (inView) {
      controls.start('visible');
    }
  }, [controls, inView]);

  const scrollToNext = () => {
    const servicesSection = document.querySelector('#services');
    if (servicesSection) {
      servicesSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3,
        delayChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.8,
        ease: "easeOut"
      }
    }
  };

  return (
    <section id="home" className="hero" ref={ref}>
      <div className="hero-background">
        <motion.div className="hero-image-container">
          {images.map((image, index) => (
            <motion.img
              key={index}
              src={image}
              alt={`Creative Work ${index + 1}`}
              className={`hero-image ${index === currentImage ? 'active' : ''}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: index === currentImage ? 1 : 0 }}
              transition={{ duration: 1 }}
            />
          ))}
        </motion.div>
        
        <div className="hero-overlay" />
        
        <div className="floating-elements">
          <motion.div
            className="floating-element element-1"
            animate={{ y: [-10, 10, -10] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div
            className="floating-element element-2"
            animate={{ y: [10, -10, 10] }}
            transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div
            className="floating-element element-3"
            animate={{ y: [-15, 15, -15] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
          />
        </div>
      </div>

      <div className="container">
        <motion.div
          className="hero-content"
          variants={containerVariants}
          initial="hidden"
          animate={controls}
        >
          <motion.div className="hero-badge" variants={itemVariants}>
            <span>🚀 Креативное агентство</span>
          </motion.div>

          <motion.h1 className="hero-title" variants={itemVariants}>
            <span className="gradient-text">Творческие решения</span>
            <br />
            для вашего бизнеса
          </motion.h1>

          <motion.p className="hero-subtitle" variants={itemVariants}>
            Мы создаем уникальные маркетинговые кампании, которые помогают брендам 
            выделяться на рынке и привлекать новых клиентов.
          </motion.p>

          <motion.div className="hero-buttons" variants={itemVariants}>
            <motion.button
              className="btn btn-primary"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                const portfolioSection = document.querySelector('#portfolio');
                if (portfolioSection) {
                  portfolioSection.scrollIntoView({ behavior: 'smooth' });
                }
              }}
            >
              Посмотреть работы
            </motion.button>
            
            <motion.button
              className="btn btn-secondary"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                const contactSection = document.querySelector('#contact');
                if (contactSection) {
                  contactSection.scrollIntoView({ behavior: 'smooth' });
                }
              }}
            >
              <FaPlay className="btn-icon" />
              Обсудить проект
            </motion.button>
          </motion.div>

          <motion.div className="hero-stats" variants={itemVariants}>
            <div className="stat-item">
              <div className="stat-icon"><FaLightbulb /></div>
              <div className="stat-number">200+</div>
              <div className="stat-label">Проектов</div>
            </div>
            <div className="stat-item">
              <div className="stat-icon"><FaChartLine /></div>
              <div className="stat-number">150%</div>
              <div className="stat-label">Рост продаж</div>
            </div>
            <div className="stat-item">
              <div className="stat-icon"><FaUsers /></div>
              <div className="stat-number">50+</div>
              <div className="stat-label">Клиентов</div>
            </div>
          </motion.div>
        </motion.div>
      </div>

      <motion.button
        className="scroll-indicator"
        onClick={scrollToNext}
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
        whileHover={{ scale: 1.1 }}
      >
        <FaArrowDown />
        <span>Прокрутите вниз</span>
      </motion.button>
    </section>
  );
};

export default Hero;
