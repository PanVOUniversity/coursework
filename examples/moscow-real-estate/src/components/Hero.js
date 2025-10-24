import React, { useEffect, useState } from 'react';
import { motion, useAnimation } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { FaArrowDown, FaPlay, FaMapMarkerAlt } from 'react-icons/fa';
import './Hero.css';

const Hero = () => {
  const [currentImage, setCurrentImage] = useState(0);
  const controls = useAnimation();
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: false
  });

  const images = [
    'https://images.unsplash.com/photo-1512453979798-5ea266f8880c?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80',
    'https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80',
    'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80'
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
    const featuresSection = document.querySelector('#features');
    if (featuresSection) {
      featuresSection.scrollIntoView({ behavior: 'smooth' });
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

  const imageVariants = {
    hidden: { scale: 1.1, opacity: 0 },
    visible: {
      scale: 1,
      opacity: 1,
      transition: {
        duration: 1.2,
        ease: "easeOut"
      }
    }
  };

  return (
    <section id="home" className="hero" ref={ref}>
      <div className="hero-background">
        <motion.div
          className="hero-image-container"
          variants={imageVariants}
          initial="hidden"
          animate={controls}
        >
          {images.map((image, index) => (
            <motion.img
              key={index}
              src={image}
              alt={`Moscow Skyline ${index + 1}`}
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
            <span>🏆 Премиальное агентство</span>
          </motion.div>

          <motion.h1 className="hero-title" variants={itemVariants}>
            <span className="gradient-text">Элитная недвижимость</span>
            <br />
            в сердце Москвы
          </motion.h1>

          <motion.p className="hero-subtitle" variants={itemVariants}>
            Найдите свой идеальный дом среди небоскребов Москва-Сити. 
            Эксклюзивные квартиры, пентхаусы и апартаменты класса люкс 
            с видом на Красную площадь.
          </motion.p>

          <motion.div className="hero-location" variants={itemVariants}>
            <FaMapMarkerAlt />
            <span>Москва, Россия</span>
          </motion.div>

          <motion.div className="hero-buttons" variants={itemVariants}>
            <motion.button
              className="btn btn-primary"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                const propertiesSection = document.querySelector('#properties');
                if (propertiesSection) {
                  propertiesSection.scrollIntoView({ behavior: 'smooth' });
                }
              }}
            >
              Посмотреть объекты
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
              Получить консультацию
            </motion.button>
          </motion.div>

          <motion.div className="hero-stats" variants={itemVariants}>
            <div className="stat-item">
              <div className="stat-number">300+</div>
              <div className="stat-label">Успешных сделок</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">12</div>
              <div className="stat-label">Лет опыта</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">95%</div>
              <div className="stat-label">Довольных клиентов</div>
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
