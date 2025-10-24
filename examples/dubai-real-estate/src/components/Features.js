import React from 'react';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { 
  FaShieldAlt, 
  FaHandshake, 
  FaChartLine, 
  FaGlobe,
  FaAward,
  FaClock
} from 'react-icons/fa';
import './Features.css';

const Features = () => {
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const features = [
    {
      icon: FaShieldAlt,
      title: 'Надежность',
      description: 'Более 15 лет опыта на рынке недвижимости Дубая',
      color: '#2d5a27'
    },
    {
      icon: FaHandshake,
      title: 'Персональный подход',
      description: 'Индивидуальная работа с каждым клиентом',
      color: '#4a7c59'
    },
    {
      icon: FaChartLine,
      title: 'Лучшие инвестиции',
      description: 'Высокая доходность и стабильный рост цен',
      color: '#6b8e6b'
    },
    {
      icon: FaGlobe,
      title: 'Международный сервис',
      description: 'Обслуживание на русском, английском и арабском языках',
      color: '#a8d5a8'
    },
    {
      icon: FaAward,
      title: 'Премиальное качество',
      description: 'Работаем только с лучшими объектами класса люкс',
      color: '#2d5a27'
    },
    {
      icon: FaClock,
      title: '24/7 Поддержка',
      description: 'Круглосуточная поддержка наших клиентов',
      color: '#4a7c59'
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
    <section id="features" className="features">
      <div className="container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Почему выбирают нас</h2>
          <p className="section-subtitle">
            Мы предоставляем исключительный сервис в мире премиальной недвижимости Дубая
          </p>
        </motion.div>

        <motion.div
          className="features-grid"
          ref={ref}
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              className="feature-card"
              variants={itemVariants}
              whileHover={{ 
                y: -10,
                transition: { duration: 0.3 }
              }}
            >
              <div className="feature-icon-container">
                <motion.div
                  className="feature-icon"
                  style={{ backgroundColor: feature.color }}
                  whileHover={{ 
                    scale: 1.1,
                    rotate: 5,
                    transition: { duration: 0.3 }
                  }}
                >
                  <feature.icon />
                </motion.div>
                <div className="feature-icon-bg" style={{ backgroundColor: feature.color }} />
              </div>
              
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
              
              <motion.div
                className="feature-hover-effect"
                style={{ backgroundColor: feature.color }}
                initial={{ scaleX: 0 }}
                whileHover={{ scaleX: 1 }}
                transition={{ duration: 0.3 }}
              />
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          className="features-cta"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          viewport={{ once: true }}
        >
          <motion.button
            className="btn btn-outline"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              const contactSection = document.querySelector('#contact');
              if (contactSection) {
                contactSection.scrollIntoView({ behavior: 'smooth' });
              }
            }}
          >
            Узнать больше о наших услугах
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
};

export default Features;
